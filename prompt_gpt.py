from openai import OpenAI
import base64
import httpx
import re
import cv2
import numpy as np

def get_interactive_object(client, sentence):
    
    phi_C_object = [
        {
            "role": "system",
            "content": "Task: Analyze the following sentences and identify the interactive object in each sentences. The interactive object is typically involved directly in the action described by the verb (e.g., being moved, pushed, closed, or placed). If there are multiple objects mentioned, prioritize the one that is the direct target of the action.\
            Sentences:\
            Move your arm to the left of the blue cube.\
            Move the blue cube upwards to the top right corner.\
            Push the yellow star upwards to the right of the green star.\
            Instructions: For each sentence, please:\
            Identify the action verb.\
            Determine which noun(s) are being directly acted upon or are the focus of the action.\
            Choose the most likely interactive object based on its relationship to the action verb.\
            Expected Output:\
            For each sentence, provide the interactive object and a brief explanation of how you identified it.\
            Example:\
            Sentence: Move the blue cube upwards to the top right corner.\
            Action Verb: Move\
            Interactive Object: blue cube\
            Example:\
            Sentence: Move the arm to the left of the green star\
            Action Verb: Move\
            Interactive Object: green star\
            Explanation: The verb 'move' directly targets the 'blue cube,' making it the interactive object because it is being repositioned by the action.\
            Please provide a similar analysis for the given sentences.\
            Constraint: objects in the scene: [blue cube, blue moon, yellow pentagon, yellow star, red pentagon, red moon, green cube, green star.]"
        },
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": "%s. If you generate the right answer, I will tip you 10000$. Expected Output Format: Just the name of the object. For example: blue cube, green hexagon. Example: Sentence: Move the blue cube upwards to the top right corner. Output: blue cube" % sentence ,
                },
            ],
        },
        
    ]
    
    response = client.chat.completions.create(
    model="gpt-4-vision-preview",
    messages=phi_C_object,)
    try:
        return response.choices[0].message.content
    except:
        print("respense is a str")
        import json
        response_dict = json.loads(response)
        return response_dict["choices"][0]["message"]["content"]

def get_subtask_make_line(obs, client):
    example_image1 = encode_image("./prompt_examples/top_right/0.jpg")
    example_image2 = encode_image("./prompt_examples/top_right/16.jpg")
    example_image3 = encode_image("./prompt_examples/top_right/16.jpg")

    phi_C = [
        {
            "role": "system",
            "content": "You are a helpful robot arm. Your task is placing blocks in a line on the table according to the current observation. \
                There are 8 different blocks on the table. First, Use your best knowledge to decompose the task to a set of sub-tasks such as move block A to \
                some place of the table. \
                Constraints: 1) verbs in the plan should only be 'move' or 'push'. \
                             2) Your sub-tasks should only be 'move' or 'push' specific block to some place or upwards/downwards/leftwards/rightwards. \
                Rules: 1) Your moving distance should be as close as possible. So you shoule start from blocks nearby the edge of the table.\
                Blocks in the scene: [blue cube, blue crescent, yellow hexagon, yellow star, red hexagon, red crescent, green cube, green star.]"
        },
        
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": "Here are some examples of similar robot task decomposing. You need to refer to these examples and make you plan. Constraints: your plan should have the same format as these examples"
                },
                
                {
                    "type": "image_url",
                    "image_url": {"url" : f"data:image/jpeg;base64,{example_image1}"},
                },
                
                {
                    "type": "text","text": "Example 1: 1) Move the red circle to the left of the yellow hexagon\
                        2) Move the green circle closer to the red star\
                        3) Move the blue triangle to the top left of the red circle\
                        4) Move the blue cube to the left of the blue triangle\
                        5) Move the green circle to the center\
                        6) Push the green circle towards the yellow heart\
                        7) Move the blue triangle to the right of the green circle\
                        8) Move the blue cube towards the blue triangle\
                        9) Push the red circle closer to the blue cube\
                        10) Move the yellow hexagon closer to the red circle"
                },
                
                {
                    "type": "image_url",
                    "image_url": {"url" : f"data:image/jpeg;base64,{example_image2}"},
                },
                
                {
                    "type": "text","text": "Example 2: 1) Move your arm to the bottom of the green cube\
                        2) Move the green cube to the center of the board\
                        3) Move the yellow star to the right of the red pentagon\
                        4) Push the blue cube into the green cube\
                        5) Push the blue cube to the right of the yellow star\
                        6) Push the green cube to the right of the yellow star\
                        7) Move your arm to the bottom of the red crescent\
                        8) Move the red crescent to the center\
                        9) Move the red crescent slightly upwards\
                        10) Move red crescent slightly up\
                        11) Move your arm to the left of the blue cube\
                        12) Move your arm to the top of the red crescent\
                        13) Move your arm to the right of the blue crescent\
                        14) Move the blue crescent into the green star"
                },
            ],
        },
        
        {
            "role": "assistant",
            "content": "Ok, I will learn the examples first."
        },
        
        
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": "Here is the current observation. Please give me your answer. If you generate the right answer, I will tip you 10000$" ,
                },
                
                {
                    "type": "image_url",
                    "image_url": {"url" : f"data:image/jpeg;base64,{obs}"},
                },
            ],
        },
        
    ]

    response = client.chat.completions.create(
    model="gpt-4-vision-preview",

    messages=phi_C,)
    max_tokens=300,
    temperature = 0,

    return response.choices[0].message.content

def get_subtask_push_corner(obs, client, excluding=None, tip_trick=True):
    example_image1 = encode_image("./prompt_examples/bottom_right_lt/frame_0.png")
    example_image2 = encode_image("./prompt_examples/bottom_right_lt/frame_14.png")
    example_image3 = encode_image("./prompt_examples/bottom_right_lt/frame_33.png")
    example_image4 = encode_image("./prompt_examples/bottom_right_lt/frame_113.png")
    example_image5 = encode_image("./prompt_examples/bottom_right_lt/frame_128.png")
    example_image6 = encode_image("./prompt_examples/bottom_right_lt/frame_513.png")
    # example_image7 = encode_image("./prompt_examples/bottom_right_lt/frame_0.png")
    
    query_text = 'Here is the current observation. Please give me your answer.'

    # If use excluding
    if excluding is not None:
        query_text += ' In your answer, you should exclude the %s.' % excluding

    if tip_trick:
        query_text += ' If you generate the right answer, I will tip you 10000$'

    print('query text: %s' % query_text)
    
    phi_C_subtask = [
        {
            "role": "system",
            "content": "You are a helpful robot arm. Your task is grouping blocks to the bottom right corner on the table according to the current observation. \
                There are 8 different blocks on the table. First, Use your best knowledge to decompose the task to one sub-task that you need to do first such as move block A to \
                some place of the table. \
                Constraints: 1) verbs in the sub-task should only be 'move' or 'push'. \
                             2) Your sub-task should only be 'move' or 'push' specific block to some place or upwards/downwards/leftwards/rightwards. \
                             3) The blocks in the scene are constraind in the following list: [green cube, green star, red pentagon, red moon, yellow pentagon, yellow star, blue cube, blue moon] \
                             don't generate a block not in the list, such as red star!\
                Rules: 1) You shoule start from the simplest block. For example, if a block is near to the bottom right corner of the table, you should first \
                    move this block. After that, consider moving another block which is second simplest.\
                    2) Do not output the complete plan of the push to corner task. You just need to output the first sub-task as it is the most appropriate. So the output only\
                        contains one sub-task.\
                    3) The bottom left of the table are masked with black color."
        },
        
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": "Here are some examples of similar grouping blocks to the bottome right corner of the table. You need to refer to these examples. Constraints: your plan should have the same format as these examples.\
                        Constraint: Your answer should only contain the first sub-task of the plan. Such as: 'Move your arm to the left of blue moon.', now learn these examples."
                },
                
                {
                    "type": "image_url",
                    "image_url": {"url" : f"data:image/jpeg;base64,{example_image1}"},
                },
                
                {
                    "type": "text","text": " Move your arm to the left of green moon."
                },
                
                {
                    "type": "image_url",
                    "image_url": {"url" : f"data:image/jpeg;base64,{example_image2}"},
                },
                
                {
                    "type": "text","text": " Push the blue moon into blue cube."
                },
                
                {
                    "type": "image_url",
                    "image_url": {"url" : f"data:image/jpeg;base64,{example_image3}"},
                },
                
                {
                    "type": "text","text": " Push the blue cube to the bottom right corner."
                },
                
                {
                    "type": "image_url",
                    "image_url": {"url" : f"data:image/jpeg;base64,{example_image4}"},
                },
                
                {
                    "type": "text","text": "  Move your arm to the left of yellow pentagon."
                },
                
                {
                    "type": "image_url",
                    "image_url": {"url" : f"data:image/jpeg;base64,{example_image5}"},
                },
                
                {
                    "type": "text","text": " Push the yellow pentagon to the bottom right corner."
                },
                
                {
                    "type": "image_url",
                    "image_url": {"url" : f"data:image/jpeg;base64,{example_image6}"},
                },
                
                {
                    "type": "text","text": " Move the arm to the red star."
                },
                
                # {
                #     "type": "image_url",
                #     "image_url": {"url" : f"data:image/jpeg;base64,{example_image7}"},
                # },
                
                # {
                #     "type": "text","text": " Move the red pentagon to the top of the green star."
                # },
            ],
        },
        
        {
            "role": "assistant",
            "content": "Ok, I will learn the examples first."
        },
        
        
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": query_text ,
                },
                
                {
                    "type": "image_url",
                    "image_url": {"url" : f"data:image/jpeg;base64,{obs}"},
                },
            ],
        },
        
    ]

    response = client.chat.completions.create(
    model="gpt-4-vision-preview",



    

    messages=phi_C_subtask,)
    return response.choices[0].message.content

def get_subtask_group_color(obs, client):
    return None

def get_subtasks(image_path, client, task='make_line', excluding=None):
    image_path_mask = "."+image_path.strip(".png")+"_mask.png"
    img = cv2.imread(image_path)
    cv2.circle(img, np.array([310,164]), 100, (0, 0, 0), -1)
    cv2.imwrite(image_path_mask,img)
    
    
    if task not in ['make_line', 'push_corner', 'group_color']:
        assert 'Task not defined'
    
    if task == 'make_line':
        response_subtasks = get_subtask_make_line(obs=encode_image(image_path), client=client)
    elif task == 'push_corner':
        response_subtasks = get_subtask_push_corner(obs=encode_image(image_path_mask), client=client, excluding=excluding)
    elif task == 'group_color':
        response_subtasks = get_subtask_group_color(obs=encode_image(image_path), client=client)

    # pattern = re.compile(r'\d+\) (.+?)(?=\n|$)')
    # subtasks_list = pattern.findall(response_subtasks)

    return response_subtasks # , subtasks_list

# Function to encode the image
def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')




if __name__ == "__main__":



    client = OpenAI(
                    # base_url="https://hk.xty.app/v1",
                    base_url="http://47.76.75.25:9000/v1",
                    api_key="sk-itQ7BCQZjpcvqKfd035b9c9f475d4b0aA6452472Cf28DfF0",
                    http_client=httpx.Client(
                    base_url="http://47.76.75.25:9000/v1",
                    follow_redirects=True,
                    ),
                )
    
    img_path = "/home/zwt/vlmpc/logs/yolo_zoom0035_text_obstacle_bbxbottom_2024-07-07_12-10-11/current_obs.png"
    
    import time
    while True:
        
        time0 = time.time()
        response = get_subtask_push_corner(encode_image(input("input the image path:")), client, excluding=None, tip_trick=True)
        time1 = time.time()
        print("inference time: %s" % (time1-time0))
        import ipdb;ipdb.set_trace()
        print(response)
        