import datetime

def current_time():
    return datetime.datetime.now().strftime('%H:%M')
    
def day_today():
    return datetime.date.today().strftime('%A')