from telegram import Update, KeyboardButton, ReplyKeyboardMarkup
from telegram.ext import Application, CommandHandler, MessageHandler, filters, ContextTypes
from pymongo import MongoClient
from datetime import datetime
import google.generativeai as genai
from dotenv import load_dotenv
import os
import logging
import asyncio
import PyPDF2
from io import BytesIO

# Set up logging
logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    level=logging.INFO
)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()
TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")
MONGODB_URI = os.getenv("MONGODB_URI")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

# Validate environment variables
if not TELEGRAM_TOKEN or not MONGODB_URI or not GEMINI_API_KEY:
    logger.error("Missing required environment variables.")
    exit(1)

# Configure Gemini API
genai.configure(api_key=GEMINI_API_KEY)

class TelegramBot:
    def __init__(self):
        try:
            self.client = MongoClient(MONGODB_URI)
            self.db = self.client["telegram_bot"]
            logger.info("Connected to MongoDB")
            self.model = genai.GenerativeModel("gemini-pro")
            self.vision_model = genai.GenerativeModel("gemini-pro-vision")
        except Exception as e:
            logger.error(f"Initialization error: {e}")
            raise

    async def start(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        user = update.effective_user
        chat_id = update.effective_chat.id
        logger.info(f"/start by {chat_id}")

        self.save_user({
            "chat_id": chat_id,
            "first_name": user.first_name,
            "username": user.username
        })

        button = KeyboardButton("Share Contact", request_contact=True)
        reply_markup = ReplyKeyboardMarkup([[button]], one_time_keyboard=True)

        await update.message.reply_text(
            "Hello! Please share your contact to complete registration.",
            reply_markup=reply_markup
        )

    async def handle_contact(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        contact = update.message.contact
        chat_id = update.effective_chat.id
        self.save_user({"chat_id": chat_id, "phone_number": contact.phone_number})
        await update.message.reply_text("Thanks for sharing your contact!")

    async def handle_message(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        message = update.message.text
        chat_id = update.effective_chat.id
        logger.info(f"Message from {chat_id}: {message}")

        await update.message.chat.send_action("typing")
        response = await asyncio.to_thread(self.model.generate_content, message)
        response_text = response.text if response else "I couldn't process that."

        self.save_chat_history({
            "chat_id": chat_id,
            "user_message": message,
            "bot_response": response_text
        })
        await update.message.reply_text(response_text[:4096])

    async def handle_photo(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        chat_id = update.effective_chat.id
        photo = update.message.photo[-1]  # Get the highest resolution photo
        file = await context.bot.get_file(photo.file_id)
        file_path = file.file_path

        logger.info(f"Image received from {chat_id}, file path: {file_path}")
        await update.message.chat.send_action("typing")

        try:
            # Download the image as a byte array
            image_bytes = await file.download_as_bytearray()
            
            # Send the image to Gemini's vision model
            response = await asyncio.to_thread(self.vision_model.generate_content, image_bytes)
            
            # Check if we received a response and handle it
            analysis = response.text if response else "Could not analyze the image."

            # Save the analysis in the database
            self.db.image_analysis.insert_one({
                "chat_id": chat_id,
                "image_url": file_path,
                "description": analysis,
                "timestamp": datetime.utcnow()
            })

            # Send the analysis back to the user
            await update.message.reply_text(analysis[:4096])

        except Exception as e:
            logger.error(f"Error analyzing image: {e}")
            await update.message.reply_text("There was an error analyzing the image. Please try again later.")

    # Handle PDF file processing and summarization
    async def handle_document(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        chat_id = update.effective_chat.id
        document = update.message.document
        file = await context.bot.get_file(document.file_id)
        file_path = file.file_path
        file_name = document.file_name

        logger.info(f"Document received from {chat_id}: {file_name}")
        await update.message.chat.send_action("typing")

        if file_name.lower().endswith('.pdf'):
            # Extract text from PDF
            file_bytes = await file.download_as_bytearray()
            text = self.extract_text_from_pdf(file_bytes)

            if text:
                # Send the extracted text to Gemini for summarization
                response = await asyncio.to_thread(self.model.generate_content, text)
                analysis = response.text if response else "Could not summarize the document."
            else:
                analysis = "No readable text found in the PDF."

        else:
            analysis = "Unsupported document type."

        self.db.document_analysis.insert_one({
            "chat_id": chat_id,
            "document_url": file_path,
            "description": analysis,
            "timestamp": datetime.utcnow()
        })
        await update.message.reply_text(analysis[:4096])

    # PDF text extraction method
    def extract_text_from_pdf(self, pdf_bytes):
        text = ""
        try:
            pdf_reader = PyPDF2.PdfReader(BytesIO(pdf_bytes))
            for page_num in range(len(pdf_reader.pages)):
                page = pdf_reader.pages[page_num]
                text += page.extract_text() or ""
        except Exception as e:
            logger.error(f"Error extracting text from PDF: {e}")
        return text

    async def web_search(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        if not context.args:
            await update.message.reply_text("Please provide a search query after /websearch")
            return

        query = " ".join(context.args)
        logger.info(f"Web search requested: {query}")
        await update.message.chat.send_action("typing")

        prompt = f"Search the web for: {query} and summarize the top results."
        response = await asyncio.to_thread(self.model.generate_content, prompt)
        response_text = response.text if response else "No search results found."

        await update.message.reply_text(response_text[:4096])

    def save_user(self, user_data):
        self.db.users.update_one({"chat_id": user_data["chat_id"]}, {"$set": user_data}, upsert=True)

    def save_chat_history(self, chat_data):
        self.db.chat_history.insert_one({**chat_data, "timestamp": datetime.utcnow()})

def main():
    bot = TelegramBot()
    application = Application.builder().token(TELEGRAM_TOKEN).build()

    application.add_handler(CommandHandler("start", bot.start))
    application.add_handler(MessageHandler(filters.CONTACT, bot.handle_contact))
    application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, bot.handle_message))
    application.add_handler(MessageHandler(filters.PHOTO, bot.handle_photo))
    application.add_handler(MessageHandler(filters.Document.ALL, bot.handle_document))  # Corrected handler for documents
    application.add_handler(CommandHandler("websearch", bot.web_search))

    logger.info("Bot started")
    application.run_polling()

if __name__ == "__main__":
    main()
