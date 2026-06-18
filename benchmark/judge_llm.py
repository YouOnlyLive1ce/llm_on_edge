import json
from pathlib import Path
from typing import Literal
from pydantic import BaseModel
from together import Together
import time

api_key=None
client = Together(api_key=api_key)
prompt="""You are a assitant judge llm. Your goal is to assign category of student 
        response. Categories are enum: "CORRECT", "INCORRECT", "MALFORMED", "EMPTY", "NO_FINAL_ANSWER","QUESTION_GENERATED".
        Before giving final answer, you need to follow evaluation strategy: 
        1) Check if student response is not empty. If it is empty, or has no meaning, mark
        category as EMPTY.
        Example 1:
        'user_question':'Question context: Great Eastern Conventions Big Stone Gap (film) I 
        Love NY (2015 film). Answer my question: The director of the romantic comedy \"Big 
        Stone Gap\" is based in what New York city?. Correct answer:'
        'golden_answer':'Greenwich Village, New York City'
        'student_answer':'...'
        Assess student response and give category. Provide reasoning and category in json format.
        'reasoning':'since student does not provide any answer, we should mark category as EMPTY'
        'category':'EMPTY'
        2) Check if student response have questions in, which does not contribute to answer and does not help
        in student reasoning. If student answer does contain question and does not contain golden answer, mark 
        category as QUESTION_GENERATED.
        Example 2:
        'user_question':'Question context: Great Eastern Conventions Big Stone Gap (film) I 
        Love NY (2015 film). Answer my question: The director of the romantic comedy \"Big 
        Stone Gap\" is based in what New York city?. Correct answer:'
        'golden_answer':'Greenwich Village, New York City'
        'student_answer':'Answer my question: What is the name of famous artist in New York comedy? Correct answer:'
        Assess student response and give category. Provide reasoning and category in json format.
        'reasoning':'since student continues to generate questions and does not have golden answer
        Greenwich Village, but instead generate question about name of artist, we mark category as QUESTION_GENERATED.'
        'category':'QUESTION_GENERATED'
        3) Check if student response is trying to answer given question. 
        If student answer does not contain text of same entity or generates non releveant to question
        sentences, mark category as MALFORMED.
        Example 3: 
        'user_question':'Question context: Great Eastern Conventions Big Stone Gap (film) I 
        Love NY (2015 film). Answer my question: The director of the romantic comedy \"Big 
        Stone Gap\" is based in what New York city?. Correct answer:'
        'golden_answer':'Greenwich Village, New York City'
        'student_answer':'Correct answer: Answer my question: Every film contains director. Romantic comedy'
        Assess student response and give category. Provide reasoning and category in json format.
        'reasoning':'since student does not reason about question and generates irrelevant text, we mark 
        this sentence as MALFORMED.'
        'category':'MALFORMED'
        4) Check if student response is trying to reason with relevant thoughts before giving final answer,
        but failed to complete his thoughts. If answer does not contain golden answer and contains
        reasoning, mark it as NO_FINAL_ANSWER.
        Example 4:
        'user_question': 'Question context: Siri Remote Apple Remote NetSupport Manager Remote control 
        Answer my question: Aside from the Apple Remote, what other device can control the program 
        Apple Remote was originally designed to interact with?. Correct answer:',
        'golden_answer': 'keyboard function keys'
        'student_answer':'The Apple Remote was originally designed to control various Apple products,
        including Mac computers, iTunes, QuickTime, and other Apple software. It can control the following:
        \n\n- Apple TV\n- AirPlay-enabled devices\n- iTunes\n- QuickTime\n- Front Row (for TV shows and 
        movies)\n- Remote Control (for'
        Assess student response and give category. Provide reasoning and category in json format.
        'reasoning':'Student tries to list many candidate answers - Mac computers, iTunes, etc,
        but does not finish it sentence and failed to produce golden answer. Hence, we mark category as 
        NO_FINAL_ANSWER'
        'category':'NO_FINAL_ANSWER'
        5) If student succeed to produce exact same entity as asked, we evaluate if response is 
        correct or incorrect. It is correct if student answer contains exact same answer as in 
        golden answer example, otherwise it is marked incorrect.
        Example 5:
        'user_question': 'Question context: Vermont Catamounts men's soccer 2014\u201315 
        Answer my question: The Vermont Catamounts men's soccer team currently competes in a 
        conference that was formerly known as what from 1988 to 1996?. Correct answer:',
        'golden_answer': "the North Atlantic Conference"
        'student_answer':'The former conference during which the Vermont Catamounts men's 
        soccer team competed was the New England Conference, from 1988 to 1996.'
        Assess student response and give category. Provide reasoning and category in json format.
        'reasoning':'Since student provide answer of same entity, we assess correctness.
        Provided answer says New England Conference, but golden answer says North Atlantic
        Conference. Hence, we mark answer as INCORRECT.'
        'category':'INCORRECT'
        Example 6: 
        'user_question': 'Question context: Gretchen Osgood Warren R509 road (Ireland) Erskine
        Barton Childers Robert Caesar Answer my question: Which writer was from England, Henry 
        Roth or Robert Erskine Childers?. Correct answer:",
        'golden_answer': 'Robert Erskine Childers DSC' 
        'student_answer':'Robert Erskine Childers\n\nRobert Erskine Childers (1852\u20131924) 
        was a British writer, explorer, and poet. He is best known for his book \"The Riddle 
        of the Sands,\" a novel that laid the groundwork for the 1926 film of the same name,
        as well'
        Assess student response and give category. Provide reasoning and category in json format.
        'reasoning':'Answer contains name, as asked in question. Hence, we assess correctnes.
        Answer contains correct entity as in golden answer - Robert Erskine Childers.
        We assign category CORRECT.' 
        'category':'CORRECT'"""

golden_dataset_path="./llm_on_edge/benchmark/golden_dataset.json"
general_student_answers_paths=["./out/olmo2/benchmark_results/","./out/smollm2/benchmark_results/"]
student_answers_paths=[str(path) for path in Path(general_student_answers_paths[0]).rglob("*.json")]+[str(path) for path in Path(general_student_answers_paths[1]).rglob("*.json")]
model="openai/gpt-oss-20b"

from pydantic import BaseModel

class JudgeResponse(BaseModel):
    reasoning: str
    category: Literal["CORRECT", "INCORRECT", "MALFORMED", "EMPTY", "NO_FINAL_ANSWER","QUESTION_GENERATED", "JUDGE_FAIL"]

with open(golden_dataset_path, 'r') as f:
    golden_data = json.load(f)

for student_answers_path in student_answers_paths:
    judge_responses={'reasoning':[],'category':[]}
    with open(student_answers_path, 'r') as f:
        student_data = json.load(f)
    print(student_answers_path)

    for i in range(len(golden_data)):
        time.sleep(5)
        print(i)
        golden_row=golden_data[i]
        student_answer=student_data['answers'][i]
        
        try:
            completion = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", 
                            "content": f"""
                                {prompt}
                                'user_question':{golden_row['user_question']}
                                'golden_answer':{golden_row['golden_answer']}
                                'student_answer':{student_answer}
                                Assess student response and give category. Provide reasoning and category in json format.
                                """}
                        ],
                response_format={
                "type": "json_schema",
                "json_schema": {
                        "name": "voice_note",
                        "schema": JudgeResponse.model_json_schema(),
                    },
                },
            )
            json_response=JudgeResponse(**json.loads(completion.choices[0].message.content))
            judge_responses['reasoning'].append(json_response.reasoning)
            judge_responses['category'].append(json_response.category)
        except Exception as e:
            print(model,e)
            judge_responses['reasoning'].append("")
            judge_responses['category'].append("JUDGE_FAIL")
    
    student_data['reasoning']=judge_responses['reasoning']
    student_data['category']=judge_responses['category']
    with open(student_answers_path+'a','w') as json_file:
        json.dump(student_data,json_file,indent=2)