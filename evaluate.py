from bert_score import score
import pandas as pd
import openai
from OpenAI_Model_QA import PDFQAWithQdrant
#from ST_Model_QA import PDFQAWithQdrant
from dotenv import load_dotenv
import os

load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")


openai.api_key = OPENAI_API_KEY

def generate_answer(qa_system, question):
    return qa_system.answer_question(question)


def evaluate_qa_system(qa_system, questions, reference_answers):
    generated_answers = [generate_answer(qa_system, question) for question in questions]
    
    P, R, F1 = score(generated_answers, reference_answers, lang="en")
    
    results_df = pd.DataFrame({
        "Question": questions,
        "Reference Answer": reference_answers,
        "Generated Answer": generated_answers,
        "Precision": P.numpy(),
        "Recall": R.numpy(),
        "F1 Score": F1.numpy()
    })
    return results_df


questions = [
    "What is the Basis of Claim (BOC) Form, and why is it important?",
    "Who qualifies as a Convention refugee under Canadian law? ",
    "What is a “person in need of protection” according to Canadian refugee policy?",
    "What happens if I do not submit my BOC Form on time?",
    "Can I submit documents in a language other than English or French?",
    "Is legal representation required for my refugee claim in Canada?",
    "What are Ready Tours, and how can they help with my refugee hearing?",
    "Who makes the final decision on my refugee claim?",
    "Will an interpreter be provided during my hearing?",
    "Can I change the official language used in my hearing after submitting my BOC Form?",
    "Are children required to attend the refugee hearing?",
    "Can I bring witnesses to my refugee hearing?",
    "What should I do if I need to change the date or time of my hearing?",
    "What does it mean if my claim is 'excluded'?",
    "What is the Refugee Appeal Division (RAD), and can all decisions be appealed?",
    "What happens if my refugee claim is allowed?",
    "What is an “abandoned claim” in the refugee process?",
    "Can I withdraw my refugee protection claim after starting the process?",
    "What are the eligibility criteria for a refugee claim to be referred to the IRB?",
    "What evidence must I provide to support my refugee claim?",
]

reference_answers = [
    "The Basis of Claim (BOC) Form is a document where you provide detailed information about yourself and the reasons for claiming refugee protection in Canada. It is crucial because the Refugee Protection Division (RPD) uses this form to assess your refugee claim",
    "A Convention refugee is someone with a well-founded fear of persecution in their home country due to race, religion, nationality, political opinion, or membership in a particular social group",
    "A person in need of protection is someone who would face a danger of torture, a threat to life, or a risk of cruel and unusual treatment or punishment if they returned to their home country",
    "If you do not submit your BOC Form by the deadline, your claim may be declared abandoned, and you will not be permitted to make another refugee protection claim in Canada",
    "No, all documents submitted to the RPD must be translated into English or French, along with a translator’s declaration verifying the translation's accuracy.",
    "No, legal representation is not required, but you may choose to have a counsel represent you. Counsel can be a lawyer, licensed immigration consultant, or a family member or friend if no fees are charged",
    "Ready Tours are informational sessions that provide an overview of the RPD hearing process, including a tour of the hearing room, preparation tips, and a chance to ask questions about the refugee process.",
    "An RPD member, a trained decision-maker in refugee protection matters, hears and decides on your refugee claim at the hearing.",
    "Yes, if you need an interpreter, the RPD will provide one at no cost. You must indicate the preferred language and dialect in your BOC Form",
    "Yes, but you must inform the RPD in writing at least 10 days before the hearing if you wish to change the official language (English or French)",
    "Children aged 12 and older claiming refugee protection must attend the hearing. Younger children may only need to attend if specifically requested by the RPD.",
    "Yes, you may bring witnesses who can provide relevant information to support your claim. You must submit witness details to the RPD at least 10 days before the hearing",
    "You can request a change, but the RPD will only approve it in exceptional circumstances. You must apply at least three working days before the hearing date.",
    "A claim is excluded if the claimant has committed serious crimes, such as war crimes, or has protection rights in another country similar to those of a citizen",
    "The RAD is a division of the IRB that handles appeals on RPD decisions. However, some decisions, such as claims declared abandoned or cases involving designated foreign nationals, cannot be appealed to the RAD.",
    "If your claim is allowed, you’ll receive a Notice of Decision and can apply for permanent residence in Canada unless the decision is appealed or overturned.",
    "A claim is considered abandoned if the claimant fails to meet key requirements, like submitting the BOC Form on time or attending hearings, leading to claim termination.",
    "Yes, you can withdraw your claim by notifying the RPD in writing. However, once withdrawn, you cannot make another refugee claim in Canada.",
    "A claim must meet basic conditions set by an officer at a port of entry or an inland office to be eligible for referral to the IRB for review.",
    "You need to provide identity documents and other supporting documents relevant to your claim, such as proof of membership in political organizations or reports on country conditions.",

]


pdf_path = "./refugee_guide.pdf"
qa_system = PDFQAWithQdrant(pdf_path)

results_df = evaluate_qa_system(qa_system, questions, reference_answers)


print("Evaluation Results:")
print(results_df)

average_f1 = results_df["F1 Score"].mean()
print(f"Average F1 Score: {average_f1:.4f}")


import streamlit as st

st.write("Evaluation Results:")
st.dataframe(results_df)

st.write(f"Average F1 Score: {average_f1:.4f}")
