from src.base.llm_model import get_hf_llm 
from src.rag.main import build_rag, InputQA, OutputQA
import gradio as gr

# Define the answer generation function
def answer_question(history, question):
    """
    This function takes the chat history and the new question as input,
    generates an answer, and updates the chat history.
    """
    # Generate answer using RAG model
    answer = genai_chain.invoke(question)
    
    # Append question and answer to history
    history.append((question, answer))
    return history

if __name__ == '__main__':
    # Initialize the LLM model
    llm = get_hf_llm()

    # Specify the document for RAG
    gen_doc = "./data_pdf/applsci-12-09820-v2.pdf"
    
    # Build the RAG chain with LLM and document
    genai_chain = build_rag(llm, data_dir=gen_doc, data_type="pdf")

    # Create a Gradio chat interface
    with gr.Blocks() as interface:
        gr.Markdown("# Chat with RAG-powered Assistant")
        
        # Initialize the chatbot component
        chatbot = gr.Chatbot(label="Chatbot")
        
        # Create input text box for user questions
        question = gr.Textbox(label="Type your question here")
        
        # Add a submit button
        submit_button = gr.Button("Send")
        
        # Define the behavior on clicking the submit button
        submit_button.click(
            fn=answer_question,             # Function to generate the answer
            inputs=[chatbot, question],     # Pass chat history and question
            outputs=chatbot                 # Output updates the chatbot history
        )

    # Launch the Gradio app
    interface.launch()
