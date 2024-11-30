# -*- coding: utf-8 -*-
"""
Main file for application

"""


import schedule
import time

from pkg.BTC_updater import update


#update()

    
print('Trading bot Status: ONLINE')
schedule.every().day.at("20:00").do(update)

while True:
    schedule.run_pending()
    time.sleep(10)