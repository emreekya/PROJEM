' MoneyPrinterTurbo WebUI - terminalden bagimsiz baslatici
' Cift tiklayarak calistir. Pencere acmaz, arka planda calisir.
Set sh = CreateObject("WScript.Shell")
Set fso = CreateObject("Scripting.FileSystemObject")
dir = fso.GetParentFolderName(WScript.ScriptFullName)
sh.CurrentDirectory = dir
sh.Environment("Process")("PYTHONPATH") = dir
py = dir & "\.venv\Scripts\python.exe"
cmd = """" & py & """ -m streamlit run "".\webui\Main.py"" --server.address=127.0.0.1 --server.port=8501 --browser.gatherUsageStats=False --server.enableCORS=True"
' 0 = pencereyi gizle, False = bitmesini bekleme
sh.Run cmd, 0, False
