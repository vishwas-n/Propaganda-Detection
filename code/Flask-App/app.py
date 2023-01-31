import json
import torch
# import TaskSC, TaskTC
from model_runnable.TaskSI import TaskSI
from model_runnable.TaskTC import TaskTC
from model_runnable.TaskSC import TaskSC

from flask import Flask, request, jsonify, render_template

app = Flask(__name__)
app.debug = True

model_si = torch.load("models/RoBERTa_Task_SI.pt", map_location=torch.device('cpu'))
model_si.eval()

model_tc = torch.load("models/RoBERTa_Task_TC.pt", map_location=torch.device('cpu'))
model_tc.eval()

model_sc = torch.load("models/RoBERTa_Task_SentClassify.pt", map_location=torch.device('cpu'))
model_sc.eval()

si = TaskSI(model_si)
tc = TaskTC(model_tc)
sc = TaskSC(model_sc)

def get_stats_map(filename):
    stats_map = {}
    with open('results/'+filename, 'r', encoding='utf-8') as fin:
        stats_map = json.load(fin)
    return stats_map

train_tech_count = get_stats_map('train_tech_count.json')
dev_tech_count = get_stats_map('dev_tech_count.json')

@app.route('/')
def display():
    print('here in display')
    return render_template("index.html",techniques ={})


@app.route('/propaganda_techiques', methods=['POST'])
def get_prop_techiques():
    type = request.args.get('type')
    print('here in prop')
    text = request.form['prop_text']
    print(text)
    if(request.form['submit_button'] == 'action1'):
        return render_template("index.html", techniques = train_tech_count)
    elif(request.form['submit_button'] == 'action5'):
        return render_template("index.html", techniques = dev_tech_count)
    elif(request.form['submit_button'] == 'action2'):
        sc_output = sc.get_prediction(text)
        return render_template("index.html", techniques = {"Sentence Level Propaganda":["Given Input: "+text, sc_output]})
    elif(request.form['submit_button'] == 'action3'):
        si_output = si.get_prediction(text)
        return render_template("index.html", techniques = {"Span Identification":["Given Input: "+text, "Identified Spans: "+si_output]})
    elif(request.form['submit_button'] == 'action4'):
        tc_output = tc.get_prediction(text)
        return render_template("index.html", techniques = {"Propaganda Classification":["Given Input: "+text, "Proganada Techiniques: "+tc_output]})


if __name__ == "__main__":
    app.run(host ="localhost", port=5002)