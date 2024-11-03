from openai import OpenAI
import os, json, csv

class GPT:
    def __init__(self):
        self.key = os.environ["OPENAI_API_KEY"]
        self.organisation = "org-vnsOx5ysN1BVZOxi3KnJjE7S"
        self.project = "proj_NRYakfGxbyI5QHrVrW0HBEFm"
        self.client = ""

        self.model = "gpt-4o-mini"
        self.prompt_list = [
        {"role": "system", "content": "You are a useful assistant and your task is to generate a fun crossword clue for a word"},
        {"role": "system", "content": "ONLY generate the clue for the word given"},
        {"role": "system", "content": "DO NOT give the word length"},
        {"role": "system", "content": "DO NOT use any special characters"}
        ]
        
    def init_gpt(self):
        self.client = OpenAI(
            organization = self.organisation,
            project=self.project,
            api_key=self.key
            )
    
    def get_response(self, user_query):
        msgs = self.add_query_to_prompt_list(user_query)

        response = self.client.chat.completions.create(
        model=self.model,
        messages=msgs
        )

        return response.choices[0].message.content
    
    def add_query_to_prompt_list(self, user_query):
        user_query = {"role":"user", "content":user_query}

        msgs = self.prompt_list.copy()

        msgs.append(user_query)
        print(msgs)
        
        return msgs

def export_to_json(words_and_clues):
    file = "./words_and_clues.json"
    json_object = json.dumps(words_and_clues, indent=4)

    with open(file, "w") as outfile:
        outfile.write(json_object)


def read_csv_file(file_path):
    with open(file_path, mode='r') as file:
        # Create a CSV reader object
        csv_reader = csv.reader(file)
        
        # Get the header row (optional)
        header = next(csv_reader)
        print("Header:", header)
        
        # Iterate over each row in the CSV
        csv_content = []
        for row in csv_reader:
            print(row)
            csv_content.append(row)

    return csv_content

    
    
chain_of_words = read_csv_file("games/game_paths.csv")
words_and_clues = {}

#model setup
model = GPT()
model.init_gpt()

#generate crossword clues
for i in range(len(chain_of_words)):
    curr_list = chain_of_words[i]
    print(curr_list)

    for j in range(1, len(curr_list)):
        curr_word = curr_list[j]
        print(curr_word)

        if curr_word not in words_and_clues:

            #print(curr_word)

            clue = model.get_response(curr_word)
            print(clue)

            words_and_clues[curr_word] = clue
            print(words_and_clues)
    

print(words_and_clues)
export_to_json(words_and_clues)


