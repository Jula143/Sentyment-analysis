import tkinter as tk
from textblob import TextBlob
from cleaner import *

def handle_click():
    user_input = text_box.get("1.0","end-1c")
    # blob = TextBlob(user_input)
    # sentiment = "positive" if blob.sentiment.polarity>0 else "negative"
    sentiment = check_sentiment(user_input)
    if sentiment=="positive":
        result_label.config(text=f'Review is {sentiment}',fg="white",bg="green")
    else:
        result_label.config(text=f'Review is {sentiment}',fg="white",bg="red")
    

window = tk.Tk()
window.title("Sentiment analysis")

text_label = tk.Label(window,text="Write or paste your review here")
text_label.pack(pady=10)

text_box = tk.Text(window,width=50,height=15)
text_box.pack(pady=10,fill=tk.BOTH, expand=True)

result_label = tk.Label(window, text="")
result_label.pack(padx=5)

check_button = tk.Button(window, text="Check sentiment analysis", command=handle_click, cursor="hand2",fg="white",bg="black")
check_button.pack(pady=10)

window.mainloop()