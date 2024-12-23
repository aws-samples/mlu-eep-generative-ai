standard_prompt_template = """

Human: Write a passage of 4 paragraphs.

The first line must contain only this title:
{title}

The first paragraph must start with sentence:
{start}

The second paragram must contain the words:
{mid}

The last paragraph must end with sentence:
{end}

Assistant:"""


cot_prompt_template = """

Human: Write a coherent passage of 4 paragraphs.

The first line must contain only this title:
{title}

The first paragraph must start with sentence:
{start}

The second paragram must contain the words:
{mid}

The last paragraph must end with sentence:
{end}

Make a plan, then write the 4 paragraphs according to the plan.

Your output should have the following format. 

Plan:
Your plan here.

Passage:
Your passage here.

Assistant:"""


plan_prompt_template = """

Human: Write a coherent passage of 4 paragraphs following this plan:
Plan:
{plan}.

The first line must contain only this title:
{title}

The first paragraph must start with sentence:
{start}

The second paragram must contain the words:
{mid}

The last paragraph must end with sentence:
{end}

Your output must have the following format. 

Plan:
Your plan here.

Passage:
Your passage here.

Assistant:"""


vote_plan_prompt_template = """

Human: Given the following choices for plans to write a coherent passage of 4 paragraphs, decide which choice is most promising.
Analyze each choice in detail.
Then conclude in the last line "The best choice is s", where s the integer ID of the choice.

Here are the choices:

{enumerated_choices}

Assistant:"""


vote_passage_prompt_template = """

Human: Given the following choices for a passage of 4 paragraphs, decide which choice is most coherent.
Analyze each choice in detail.
Then conclude in the last line "The best choice is s", where s the integer id of the choice.

Here are the choices:

{enumerated_choices}

Assistant:"""


score_coherency_prompt_template = """

Human: Analyze the following passage for coherency. 
At the last line conclude "Thus the coherency score is s", where s is an integer from 1 to 10.
1 indicates that the coherency of the passage is poor. 
10 indicates great coherency of the passage.

Passage:

{passage}

Assistant:
"""


compare_prompt_template = """

Human: Compare the coherency of the two passages below. 
If Passage 1 is clearly more coherent, conclude in the last line "The more coherent passage is 1".
If Passage 2 is clearly more coherent, conclude in the last line "The more coherent passage is 2".
If both passages are roughly similarly coherent, conclude in the last line "The two passages are similarly coherent".

Passage 1:
{passage1}

Passage 2:
{passage2}

Assistant:"""
