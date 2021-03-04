import tkinter as tk
from chat_bot import get_response, bot_name

class Chat:
    def __init__(self):
        self.window = tk.Tk()
        self._setup_window()
        
    def _setup_window(self):
        self.window.title("Bot Maartenok")
        self.window.resizable(width=False, height=False)
        _width, _height = 400, 550
        self.window.configure(width=_width, height=_height, bg='grey')
        
        # Center the window
        screen_width = self.window.winfo_screenwidth() 
        screen_height = self.window.winfo_screenheight()
        # For left-alling
        left = (screen_width / 2) - (_width / 2)  
        # For right-allign
        top = (screen_height / 2) - (_height /2)  
        # For top and bottom
        self.window.geometry('%dx%d+%d+%d' % (_width, _height,
            left, top))
        
        top_label = tk.Label(self.window, bg='grey', fg='white',
            text='Maartenok', pady=6, font='12',)
        top_label.place(relwidth=1)
        
        divide_line = tk.Label(self.window, width=400, bg='green')
        divide_line.place(relwidth=1, rely=0.06, relheight=0.012)
        
        # text instance variable
        self.text_widget = tk.Text(self.window, width=20, height=2,
            bg='grey13', padx=5, pady=5, fg='white', wrap='word',
            font='Courier 12')
        self.text_widget.place(relheight=0.745, relwidth=0.97, rely=0.07)
        self.text_widget.configure(state=tk.DISABLED, cursor='arrow')
        
        scrollbar = tk.Scrollbar(self.window)
        scrollbar.place(relheight=0.744, relx=0.97, relwidth=0.03, rely=0.07)
        scrollbar.configure(command=self.text_widget.yview)
        self.text_widget.config(yscrollcommand=scrollbar.set)
        
        bottom_label = tk.Label(self.window, bg='grey', height=80)
        bottom_label.place(relwidth=1, rely=0.825)
        
        # message instance variable
        self.message_entry = tk.Entry(bottom_label, bg='white', font='12')
        self.message_entry.place(relwidth=0.75, relheight=0.05, rely=0.012,
            relx=0.011)
        self.message_entry.focus() # focus entry widget when app launched
        self.message_entry.bind("<Return>", self._on_return_pressed)
        
        send_button = tk.Button(bottom_label, text='Send', width=20,
            bg='green', command=lambda: self._on_return_pressed(None), 
            font='12',)
        send_button.place(relx=0.78, rely=0.012, relheight=0.05, relwidth=0.20)
        
    def _on_return_pressed(self, event=None):
        global _message
        _message = self.message_entry.get()
        self._insert_message(_message, 'You')
        self.window.after(1500, self._insert_answer) # delay the answer
        
    def _insert_message(self, _message, sender):
        '''Insert message into text widget'''
        if not _message:
            return
        
        self.message_entry.delete(0, tk.END)
        message1 = f"{sender}: {_message}\n\n"
        self.text_widget.configure(state=tk.NORMAL)
        self.text_widget.insert(tk.END, message1)
        self.text_widget.configure(state=tk.DISABLED)
        self.text_widget.see(tk.END) # always scroll down to see last message
        
    def _insert_answer(self):
        '''Insert answer from chat_bot'''
        self.text_widget.tag_config('bot', foreground='green')
        message2 = f"{bot_name}: {get_response(_message)}\n\n"
        self.text_widget.configure(state=tk.NORMAL)
        self.text_widget.insert(tk.END, message2, 'bot')
        self.text_widget.configure(state=tk.DISABLED)
        self.text_widget.see(tk.END)
        
    def run(self):
        self.window.mainloop()
        
        
if __name__ == '__main__':
    chat = Chat()
    chat.run()