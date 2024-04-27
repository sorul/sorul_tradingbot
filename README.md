# SORUL TRADINGBOT - Forex bot in Metatrader 5 🤖📈

## What is this project about
This project uses the ⭐🚀 [tradeo](https://github.com/sorul/tradeo) 🚀⭐ library to build a forex bot using MetaTrader 5.

I've used [pi-apps](https://github.com/Botspot/pi-apps) repo to install Wine 🍷 into my Raspberry Pi 4. Later, I installed this [MetaTrader](https://download.mql5.com/cdn/web/metaquotes.software.corp/mt4/mt4setup.exe?utm_source=www.metatrader4.com&utm_campaign=download) executable using Wine.


## MetaTrader config

Once MetaTrader is installed, download [mt_tb_expert](https://github.com/sorul/tradeo/blob/master/tradeo/mt_tb_expert.mq5) file and put it in the proper directory. This directory is usually: *MetaTrader/MQL5/Experts/Advisors/mt_tb_expert.mq5*.

Open the MetaTrader and do the login. Add the symbols in the timeframe you are going to use (in my case it would be 5 minutes). There is the option for the bot to open them automatically, but I do not recommend it.

Activate the expert in any symbol chart, it does not matter which chart you use. But only in one of them.


## How I execute the project

I edit de crontab (crontab -e):

```console
@reboot /usr/bin/env bash /home/pi/sorul_tradingbot/scripts/execute_mt5.sh

*/5 * * * 0-5 /home/pi/sorul_tradingbot/scripts/main_forex.sh >> /tmp/crontab_script_log.txt 2>&1
```
