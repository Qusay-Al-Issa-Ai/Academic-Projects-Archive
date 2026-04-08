@echo off
:: تفعيل بيئة أناكوندا أولاً (تأكد من المسار)
call "C:\Users\C E C\anaconda3\Scripts\activate.bat" base

cd "C:\Users\C E C\Desktop\skin_clasification2"
:: ثم تشغيل ستريمليت
streamlit run interface_file_streamlit.py
pause