import os
import re
import fitz
import docx2txt
from transformers import BertTokenizer, BertModel
import torch
from bs4 import BeautifulSoup

device = 'cuda' if torch.cuda.is_available() else 'cpu'
tokenizer = BertTokenizer.from_pretrained('shibing624/text2vec-base-chinese')
model = BertModel.from_pretrained('shibing624/text2vec-base-chinese')

class PromptGenerator:
    def __init__(self, knowledge_dir = 'documents', k = 5, chunk_length = 128):
        self.knowledge_dir = knowledge_dir  # Path to the knowledge base
        self.k = k  # The number of most related paragraphs to be included in the prompt
        self.chunk_length = chunk_length  # Length of each chunk of text
        self._read_paragraphs()


    # Read all pdf, docx and html files in the given directory and convert them into plain text
    def _read_files(self, directory):
        for root, dirs, files in os.walk(directory):
            for filename in files:
                if filename.endswith('.pdf') or filename.endswith(
                        '.docx') or filename.endswith('.html'):
                    filepath = os.path.join(root, filename)

                    with open(filepath, 'rb') as f:
                        if filename.endswith('.pdf'):
                            text = ""
                            pdfdoc = fitz.open(stream=f.read(), filetype="pdf")

                            for page in pdfdoc:
                                text += page.get_text()

                        elif filename.endswith('.docx'):
                            text = docx2txt.process(f)
                        else:
                            soup = BeautifulSoup(f, 'html.parser')
                            text = soup.get_text()
                        text = text.replace('\n', '')
                        yield text


    # Split the text into chunks with a length of chunk_length and clean their format
    def _split_text(self, text, chunk_length):
        keep_chars = r'[\u4e00-\u9fa5a-zA-Z0-9,.!?，。？\s]+'
        cleaned_text = re.sub(r'\s{2,}', ' ', text)
        cleaned_text = re.findall(keep_chars, cleaned_text)
        cleaned_text = ''.join(cleaned_text)

        text_chunks = [
            cleaned_text[i:i + chunk_length]
            for i in range(0, len(cleaned_text), chunk_length)
        ]

        return text_chunks


    # Paragraph embedding
    def _embed_paragraphs(self, paragraphs, chunk_length):
        inputs = tokenizer(paragraphs,
                        padding='max_length',
                        max_length=chunk_length,
                        truncation=True,
                        return_tensors='pt')
        with torch.no_grad():
            model_output = model(**inputs)

        attention_mask = inputs['attention_mask']

        token_embeddings = model_output[0]
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(
            token_embeddings.size()).float()
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(
            input_mask_expanded.sum(1), min=1e-9)


    # Calculate the similarity between the question embedding and the embedding of each paragraph,
    # and find the k paragraphs with the highest similarity
    def _find_similar_paragraphs(self, embedded_paragraphs, embedded_question, k):
        similarity_scores = torch.matmul(embedded_paragraphs,
                                        embedded_question.T).squeeze().numpy()

        top_k_indices = similarity_scores.argsort()[::-1][:k]
        return top_k_indices


    # Prompt generation
    def _generate_prmopt(self, paragraphs, top_k_indices, question):
        input_text = 'You are given the following information.\n'
        for i in top_k_indices:
            input_text += '- {}\n'.format(paragraphs[i])
        input_text += '\nYou are now asked to respond to 「{}」 according to the given information.\n' \
        'If you can\'t answer the question with the given information, you can ignore those given information and answer freely. Response in the same language as the question.'.format(question[0])
        
        return input_text
    

    def _read_paragraphs(self):
        self._text_data = list(self._read_files(self.knowledge_dir))
        self._paragraphs = []
        for text in self._text_data:
            self._paragraphs += self._split_text(text, self.chunk_length)
        # print(self._paragraphs)

        self._embedded_paragraphs = self._embed_paragraphs(self._paragraphs, self.chunk_length)
    

    def get_prompt(self, question):
        embedded_question = self._embed_paragraphs(question, self.chunk_length)
        top_k_indices = self._find_similar_paragraphs(self._embedded_paragraphs,
                                                embedded_question, self.k)
        
        prompt = self._generate_prmopt(self._paragraphs, top_k_indices, question)
        return prompt


if __name__ == '__main__':
    promptGenerator = PromptGenerator()
    prompt = promptGenerator.get_prompt(['Tell me about the Wei Lun Hall'])
    print(prompt)
