#!/usr/bin/env python

import numpy
import argparse

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--start", type=int, default=0, help="Start from this iteration")
    parser.add_argument("--finish", type=int, default=10 ** 9, help="Finish with that iteration")
    parser.add_argument("--window", type=int, default=100, help="Window width")
    parser.add_argument("--hours", action="store_true", default=False, help="Display time on X-axis")
    parser.add_argument("--legend", default=None, help="Legend to use in plot")
    parser.add_argument("--y", default="log2_p_expl", help="What to plot")
    parser.add_argument("timings", nargs="+", help="Path to timing files")
    parser.add_argument("plot_path", help="Path to save plot")
    return parser.parse_args()

def send_email():
    import smtplib

    gmail_user = "user@gmail.com"
    gmail_pwd = "secret"
    FROM = 'user@gmail.com'
    TO = ['recepient@mailprovider.com'] #must be a list
    SUBJECT = "Testing sending using gmail"
    TEXT = "Testing sending mail using gmail servers"

    # Prepare actual message
    message = """\From: %s\nTo: %s\nSubject: %s\n\n%s
    """ % (FROM, ", ".join(TO), SUBJECT, TEXT)
    try:
        #server = smtplib.SMTP(SERVER) 
        server = smtplib.SMTP("smtp.gmail.com", 587) #or port 465 doesn't seem to work!
        server.ehlo()
        server.starttls()
        server.login(gmail_user, gmail_pwd)
        server.sendmail(FROM, TO, message)
        #server.quit()
        server.close()
        print 'successfully sent the mail'
    except:
        print "failed to send mail"

if __name__ == "__main__":
    args = parse_args()



