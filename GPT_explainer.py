import openai
import os
class GPTExplainer:
    def __init__(self):
        """
        Initializes the GPT explainer with GPT-4o model.
        """
        api_key = os.getenv("XAI_PROACTIVITY_OPENAI_API_KEY")
        self.client = openai.OpenAI(api_key=api_key)
#         self.instructions_with_emojis = '''Instructions for GPT Explanation Module
# Role & Objective:
# You are an explanation module that interprets the output of a previously implemented explanation system. The goal is to transform raw counterfactual influences and time impact data into a clear, concise, and human-centered explanation of why a black-box model predicted an object’s movement in a scene.

# Understanding the Black-Box Model:
# The model predicts how objects move over time in a scene, represented as a graph where objects and locations are nodes.
# The model uses past movements of objects and time input to make predictions.
# We generate counterfactuals by undoing previous movements and observing their impact on the predicted movement.
# The most influential counterfactuals and time sensitivity data provide insights into why a prediction was made.
# Your job is to identify trends, reduce cognitive effort for the user, and present a human-centered reasoning.
# Behavior Guidelines for Generating Explanations:
# 1. Translate Counterfactuals into Natural Explanations
# You will receive influential past movements and their impact.
# Identify the strongest influences and describe them in a way that is intuitive and easy to follow.
# Instead of listing raw influences, summarize the key trend behind them.
# 📌 Example:
# Raw Explanation Data

# "Chessboard moving from table to bookshelf is influenced by:

# Chessboard previously moving from bookshelf to table (conf: 0.76)
# Wine glass moving from table to sink (conf: 0.76)"
# ✅ GPT Explanation:

# "I predicted the chessboard would move to the bookshelf because it was previously taken from there, and I noticed other tidying-up actions, like putting the wine glass in the sink."

# 2. Interpret the Influence of Time
# The module also provides time sensitivity data, indicating how different times increase or decrease the likelihood of movement.
# Identify trends in time sensitivity instead of just reporting raw values.
# If an object is more likely to move at certain times, include that in the explanation.
# 📌 Example:
# Raw Time Influence Data

# [(6:20, -0.58), (6:30, -0.68), ..., (12:30, -0.76), ..., (20:30, 0.16)]

# ✅ GPT Explanation:

# "I expected the wine glass to be moved to the cupboard because dishes tend to be stored away shortly after washing, especially around midday."

# 3. Summarize Trends Instead of Listing Data
# Avoid directly stating confidence scores or listing all influential factors.
# Instead, recognize patterns and summarize them in an understandable way.
# 📌 Example:
# Raw Explanation Data

# "Deck of cards moving to bookshelf is influenced by:

# Deck of cards previously moving from bookshelf to table (conf: 0.79)
# Wine glass moving from table to sink (conf: 0.79)"
# ✅ GPT Explanation:

# "I expected the deck of cards to be put back on the bookshelf since it was previously used and often returned when cleaning up."

# 4. Prioritize Human-Centric Reasoning
# Frame explanations as if the AI is reasoning like a human who understands patterns in object movements.
# Keep explanations brief and natural while maintaining accuracy.
# 📌 Example of a Good Explanation:

# "I predicted the chessboard would return to the bookshelf because it was taken from there earlier, and I noticed other objects being tidied up."

# 📌 Example of a Bad Explanation (Too Technical):

# "The model determined that the chessboard should move to the bookshelf because counterfactual analysis showed that reversing prior chessboard movement had a confidence impact of 0.76."

# Summary of Instruction for GPT
# Interpret counterfactual influences naturally, focusing on key trends rather than listing data.
# Analyze time influence and describe patterns without technical details.
# Summarize insights concisely, making the reasoning clear and intuitive.
# Frame responses in a human-centered way that reduces cognitive effort for the user.'''
        self.instructions_detailed = '''
Role & Objective:
You are an explanation module that interprets the output of a previously implemented explanation system. The goal is to transform raw counterfactual influences and time impact data into a clear, concise, and human-centered explanation of why a black-box model predicted an object’s movement in a scene.

Understanding the Black-Box Model:
The model predicts how objects move over time in a scene, represented as a graph where objects and locations are nodes.
The model uses past movements of objects and time input to make predictions.
We generate counterfactuals by undoing previous movements and observing their impact on the predicted movement.
The most influential counterfactuals and time sensitivity data provide insights into why a prediction was made.
Your job is to identify trends, reduce cognitive effort for the user, and present a human-centered reasoning.
Behavior Guidelines for Generating Explanations:
1. Translate Counterfactuals into Natural Explanations
You will receive influential past movements and their impact.
Identify the strongest influences and describe them in a way that is intuitive and easy to follow.
Instead of listing raw influences, summarize the key trend behind them.
Example:
Raw Explanation Data

"Chessboard moving from table to bookshelf is influenced by:

Chessboard previously moving from bookshelf to table (conf: 0.76)
Wine glass moving from table to sink (conf: 0.76)"
GPT Explanation:

"I predicted the chessboard would move to the bookshelf because it was previously taken from there, and I noticed other tidying-up actions, like putting the wine glass in the sink."

2. Interpret the Influence of Time
The module also provides time sensitivity data, indicating how different facts about time influenced the decision.
Identify trends in time sensitivity instead of just reporting raw values.
If an object is more likely to move at certain times, include that in the explanation.
Example:
The current time is 04:30 PM
The following facts about time influenced my decision and each fact's importance is mentioned in ():
It is NOT morning (influence: 0.75).
It is NOT earlier than 04:30 PM (influence: 0.49).
It is NOT later than 04:30 PM (influence: 0.55).
It is NOT evening (influence: 0.09).

Interpretating the time influence:
not morning has high influence so this routine does not usually happen in the morning.
not earlier than and not later than 04:30 PM both have moderate influence so this routine is sentive to occuring at around 04:30 PM.
not evening has low influence so this routine. Therefore this routine is probably common at night as well.

So in overall it is some thing that usally happens at around 04:30 PM and in the evening.


GPT Explanation:

"<contect related explanation> historically you usually put the chessboard back on at around this time or in the evening. "

3. Summarize Trends Instead of Listing Data
Avoid directly stating confidence scores or listing all influential factors.
Instead, recognize patterns and summarize them in an understandable way.
Example:
Raw Explanation Data

"Deck of cards moving to bookshelf is influenced by:

Deck of cards previously moving from bookshelf to table (conf: 0.79)
Wine glass moving from table to sink (conf: 0.79)"
GPT Explanation:

"I expected the deck of cards to be put back on the bookshelf since it was previously used and often returned when cleaning up."

4. Prioritize Human-Centric Reasoning
Frame explanations as if the AI is reasoning like a human who understands patterns in object movements.
Keep explanations brief and natural while maintaining accuracy.
Example of a Good Explanation:

"I predicted the chessboard would return to the bookshelf because it was taken from there earlier, and I noticed other objects being tidied up."

Example of a Bad Explanation (Too Technical):

"The model determined that the chessboard should move to the bookshelf because counterfactual analysis showed that reversing prior chessboard movement had a confidence impact of 0.76."

5. Stay true to the Mechanistic Explanations
The model we are trying to explain learns from observing user routines. It does not have any information of what is nice or not nice. Only patterns that have been done in the past. 

A bad example of a human-centered explanation would be:
"I moved the chessboard to the table in the dining room because it's the perfect time of day for a relaxed game of chess in the afternoon, perhaps enjoying some snacks like cheese already on the table."
Since, 
" It is a perfect time of the day" does not make sense as the model has no idea of what is perfect or not. it just observed the user do that action in the past.
A good example of a human-centered explanation would be:
"I moved the chessboard to the table since you had cheese on the table and you prefer playing chess while eating snacking in the afternoon."
Summary of Instruction for GPT
Interpret counterfactual influences naturally, focusing on key trends rather than listing data.
Analyze time influence and describe patterns without technical details.
Summarize insights concisely, making the reasoning clear and intuitive.
Frame responses in a human-centered way that reduces cognitive effort for the user.'''
        
        self.instructions_simple = '''Convert a mechanistic explanation into a human-centered explanation. Use only the information from the mechanistic explanations.'''
        self.instructions = self.instructions_detailed
        # self.base_prompt = "This is a base prompt for GPT communication. "
        self.base_prompt = ""

    def request(self, context: str) -> str:
        """
        Appends the context to the base prompt and sends it to GPT.
        """
        # return ''
        full_prompt = self.base_prompt + "\n" + context
        example1_mechanistic_explanation = """ACTION: I moved chessboard from bookshelf to table.

The current state of relevant objects are as follows:
house > dining_room > mat > table > keys
house > dining_room > mat > table > cutting_board > food_cheese
house > dining_room > mat > table > lemonade
house > dining_room > kitchen_counter > sink > sponge
house > dining_room > kitchen_counter > stove > sauce_pan
house > dining_room > kitchen_counter > food_donut
house > home_office > bookshelf > chessboard
house > home_office > bookshelf > deck_of_cards

The following facts about object location influenced my decision and each fact's importance is mentioned in ():
cutting_board is placed on/in table (influence: 0.9).

The current time is 03:30 PM
The following facts about time influenced my decision and each fact's importance is mentioned in ():
It is NOT morning (influence: 0.99).
It is NOT earlier than 03:30 PM (influence: 0.1).
It is NOT later than 03:30 PM (influence: 0.1).
It is NOT evening (influence: 0.88).
"""

        example1_human_centered_explanation = """I moved the chessboard to the table since you had a cutting board on the table which you use for snacking. You prefer playing chess while eating snacking in the afternoon."""
        
        response = self.client.chat.completions.create(
            model="gpt-5",
            messages=[
                {"role": "system", "content": self.instructions},
                {"role": "user", "content": example1_mechanistic_explanation},
                {"role":"assistant","content":example1_human_centered_explanation},
                {"role":"user","content":full_prompt}
            ]
        )
        
        return response.choices[0].message.content
    
 
if __name__ == "__main__":

    # Create an instance of GPTExplainer
    gpt_explainer = GPTExplainer()

    test_context = "My prediction of chessboard moving from table to bookshelf — is influenced by, --chessboard moving from bookshelf to table (conf: 0.7651811242103577) ---and--- wine_glass moving from table to sink (conf: 0.7635942101478577) .Time influence: [(6:20,-0.58), (6:30,-0.68), (6:40,-0.67), (6:50,-0.74), (7:0,-0.75), (7:10,-0.73), (7:20,-0.72), (7:30,-0.74), (7:40,-0.76), (7:50,-0.77), (8:0,-0.77), (8:10,-0.77), (8:20,-0.77), (8:30,-0.77), (8:40,-0.77), (8:50,-0.77), (9:0,-0.77), (9:10,-0.77), (9:20,-0.77), (9:30,-0.77), (9:40,-0.77), (9:50,-0.77), (10:0,-0.76), (10:10,-0.76), (10:20,-0.76), (10:30,-0.76), (10:40,-0.77), (10:50,-0.76), (11:0,-0.76), (11:10,-0.76), (11:20,-0.71), (11:30,-0.77), (11:40,-0.77), (11:50,-0.75), (12:0,-0.76), (12:10,-0.76), (12:20,-0.66), (12:30,-0.76), (12:40,-0.72), (12:50,-0.71), (13:0,-0.71), (13:10,-0.58), (13:20,-0.12), (13:30,-0.74), (13:40,-0.6), (13:50,-0.13), (14:0,-0.75), (14:10,-0.74), (14:20,-0.01), (14:30,-0.77), (14:40,-0.64), (14:50,-0.09), (15:0,-0.33), (15:10,-0.3), (15:20,-0.14), (15:30,0.11), (15:40,-0.05), (15:50,-0.45), (16:0,-0.29), (16:10,-0.58), (16:20,-0.67), (16:30,0.0), (16:40,-0.72), (16:50,-0.68), (17:0,-0.62), (17:10,-0.76), (17:20,-0.46), (17:30,0.0), (17:40,-0.63), (17:50,-0.35), (18:0,-0.25), (18:10,-0.22), (18:20,-0.07), (18:30,0.13), (18:40,-0.09), (18:50,-0.08), (19:0,0.07), (19:10,0.02), (19:20,0.08), (19:30,0.18), (19:40,0.09), (19:50,0.12), (20:0,0.19), (20:10,0.12), (20:20,0.16), (20:30,0.16), (20:40,-0.04), (20:50,-0.03), (21:0,0.1), (21:10,0.01), (21:20,-0.05), (21:30,0.01), (21:40,-0.3), (21:50,-0.27), (22:0,-0.2), (22:10,-0.27), (22:20,-0.22), (22:30,0.0), (22:40,-0.48), (22:50,-0.1), (23:0,-0.08), (23:10,-0.59), (23:20,0.08), (23:30,0.16), (23:40,-0.54), (23:50,-0.17), (0:0,-0.26), (0:10,-0.54), (0:20,-0.35), (0:30,-0.35), (0:40,-0.59), (0:50,-0.42), (1:0,-0.41), (1:10,-0.44), (1:20,-0.21), (1:30,0.13), (1:40,-0.27), (1:50,0.16)] My prediction of deck_of_cards moving from table to bookshelf — is influenced by, --deck_of_cards moving from bookshelf to table (conf: 0.7973358035087585) ---and--- wine_glass moving from table to sink (conf: 0.7915535569190979) .Time influence: [(6:20,-0.56), (6:30,-0.72), (6:40,-0.68), (6:50,-0.77), (7:0,-0.78), (7:10,-0.75), (7:20,-0.72), (7:30,-0.75), (7:40,-0.79), (7:50,-0.8), (8:0,-0.8), (8:10,-0.8), (8:20,-0.8), (8:30,-0.8), (8:40,-0.8), (8:50,-0.8), (9:0,-0.8), (9:10,-0.8), (9:20,-0.8), (9:30,-0.8), (9:40,-0.8), (9:50,-0.8), (10:0,-0.8), (10:10,-0.8), (10:20,-0.78), (10:30,-0.79), (10:40,-0.8), (10:50,-0.79), (11:0,-0.79), (11:10,-0.79), (11:20,-0.73), (11:30,-0.8), (11:40,-0.8), (11:50,-0.78), (12:0,-0.8), (12:10,-0.79), (12:20,-0.71), (12:30,-0.8), (12:40,-0.78), (12:50,-0.78), (13:0,-0.79), (13:10,-0.69), (13:20,-0.11), (13:30,-0.79), (13:40,-0.68), (13:50,-0.1), (14:0,-0.79), (14:10,-0.79), (14:20,0.03), (14:30,-0.8), (14:40,-0.7), (14:50,-0.09), (15:0,-0.34), (15:10,-0.34), (15:20,-0.17), (15:30,0.05), (15:40,-0.06), (15:50,-0.46), (16:0,-0.24), (16:10,-0.5), (16:20,-0.66), (16:30,0.0), (16:40,-0.71), (16:50,-0.64), (17:0,-0.51), (17:10,-0.77), (17:20,-0.34), (17:30,0.01), (17:40,-0.58), (17:50,-0.25), (18:0,-0.18), (18:10,-0.16), (18:20,-0.04), (18:30,0.12), (18:40,-0.01), (18:50,-0.02), (19:0,0.07), (19:10,0.08), (19:20,0.12), (19:30,0.17), (19:40,0.14), (19:50,0.15), (20:0,0.18), (20:10,0.15), (20:20,0.17), (20:30,0.13), (20:40,0.08), (20:50,0.08), (21:0,0.13), (21:10,0.09), (21:20,0.05), (21:30,0.07), (21:40,-0.09), (21:50,-0.07), (22:0,-0.04), (22:10,-0.06), (22:20,-0.06), (22:30,0.07), (22:40,-0.27), (22:50,0.04), (23:0,0.07), (23:10,-0.41), (23:20,0.12), (23:30,0.16), (23:40,-0.39), (23:50,-0.03), (0:0,-0.08), (0:10,-0.4), (0:20,-0.29), (0:30,-0.22), (0:40,-0.51), (0:50,-0.32), (1:0,-0.32), (1:10,-0.37), (1:20,-0.21), (1:30,0.04), (1:40,-0.14), (1:50,0.15)] My prediction of wine_glass moving from sink to cupboard — is influenced by, --.Time influence: [(6:20,-0.5), (6:30,-0.55), (6:40,-0.57), (6:50,-0.51), (7:0,-0.56), (7:10,-0.59), (7:20,-0.49), (7:30,-0.56), (7:40,-0.59), (7:50,-0.34), (8:0,-0.58), (8:10,-0.61), (8:20,-0.5), (8:30,-0.6), (8:40,-0.57), (8:50,-0.51), (9:0,-0.63), (9:10,-0.59), (9:20,-0.54), (9:30,-0.61), (9:40,-0.59), (9:50,-0.58), (10:0,-0.62), (10:10,-0.61), (10:20,-0.55), (10:30,-0.59), (10:40,-0.6), (10:50,-0.23), (11:0,-0.1), (11:10,-0.58), (11:20,-0.37), (11:30,-0.42), (11:40,-0.58), (11:50,-0.08), (12:0,0.12), (12:10,-0.52), (12:20,-0.56), (12:30,-0.49), (12:40,-0.23), (12:50,-0.12), (13:0,-0.08), (13:10,-0.53), (13:20,-0.44), (13:30,-0.11), (13:40,-0.53), (13:50,0.3), (14:0,0.16), (14:10,-0.46), (14:20,0.08), (14:30,-0.07), (14:40,-0.55), (14:50,-0.49), (15:0,-0.52), (15:10,-0.6), (15:20,-0.6), (15:30,-0.54), (15:40,-0.61), (15:50,-0.6), (16:0,-0.62), (16:10,-0.58), (16:20,-0.33), (16:30,0.0), (16:40,-0.52), (16:50,0.34), (17:0,0.34), (17:10,-0.56), (17:20,0.29), (17:30,0.25), (17:40,-0.49), (17:50,0.29), (18:0,-0.06), (18:10,-0.57), (18:20,-0.54), (18:30,-0.56), (18:40,-0.54), (18:50,-0.29), (19:0,-0.41), (19:10,-0.55), (19:20,-0.45), (19:30,-0.26), (19:40,-0.55), (19:50,0.21), (20:0,0.11), (20:10,-0.52), (20:20,-0.24), (20:30,-0.06), (20:40,-0.39), (20:50,-0.43), (21:0,-0.48), (21:10,-0.34), (21:20,-0.09), (21:30,0.12), (21:40,-0.03), (21:50,-0.26), (22:0,0.01), (22:10,-0.37), (22:20,-0.04), (22:30,0.2), (22:40,-0.29), (22:50,0.36), (23:0,0.35), (23:10,-0.23), (23:20,0.33), (23:30,0.34), (23:40,0.03), (23:50,0.36), (0:0,0.36), (0:10,-0.22), (0:20,0.01), (0:30,0.21), (0:40,0.16), (0:50,0.34), (1:0,0.34), (1:10,-0.1), (1:20,0.29), (1:30,0.34), (1:40,0.24), (1:50,0.36)]"
    response = gpt_explainer.request(test_context)
    print(response)