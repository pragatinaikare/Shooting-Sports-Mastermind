from flask import Flask, request, render_template

from shooting_llm import *

application = Flask(__name__)

app = application

llm = LLMLangchain()

## route for home


@app.route('/')
def index():
    return render_template("Shooting.html")

@app.route('/llmtrigger', methods=['POST', 'GET'])
def reports():
    if request.method == 'POST':
        if 'message' in request.form:
            query = request.form['message']

            # default_prompt = HumanMessage
            # content=query
            # )

            # llm.messages.append(default_prompt)

            # default_res = str(llm.chat(llm.messages).content)llm_res=default_res,


            prompt = llm.augment_prompt(query=query)
            prompt = HumanMessage(
            content=prompt
            )
            # llm.messages = []

            llm.messages.append(prompt)

            res = llm.chat(llm.messages)
            response= {"message":res.content}
            print("#############----Output----###########################")
            return(response)




if __name__ == "__main__":
    #app.run(host="0.0.0.0", port=8000, debug=True)
    #app.run(host="0.0.0.0", port=5000, debug=False)
    
    app.run()
