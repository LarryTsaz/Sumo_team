# Sumo_team

使用 SUMO 與深度強化學習進行交通號誌控制的實驗專案。

## 系統需求

- Python 3.12（目前測試版本為 3.12.10）
- Eclipse SUMO 1.27.x
- Git

SUMO 是獨立的模擬器，不能只靠 `pip` 完成安裝。Windows 建議使用
[SUMO 官方安裝程式](https://sumo.dlr.de/docs/Installing/index.html)。安裝後重新開啟終端機，並確認：

```powershell
sumo --version
$env:SUMO_HOME
```

若 `$env:SUMO_HOME` 沒有值，請將 Windows 環境變數 `SUMO_HOME` 設為 SUMO
安裝目錄（該目錄下應有 `bin` 與 `tools`），並將 `%SUMO_HOME%\bin` 加入 `Path`。

## 在新電腦安裝

以下指令請在 PowerShell 執行：

```powershell
git clone https://github.com/LarryTsaz/Sumo_team.git
cd Sumo_team
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

如果 PowerShell 阻止啟用虛擬環境，可直接使用虛擬環境中的 Python：

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

## 執行

在專案根目錄執行訓練：

```powershell
python -m traffic_rl.main
```

執行已儲存模型的評估：

```powershell
python -m traffic_rl.evaluate_dqn
```

模型檔 `ocba_dqn_model.pth` 與 `vanilla_dqn_model.pth` 放在專案根目錄。
路網檔案則位於 `traffic_rl/nets/`。程式使用相對路徑，因此專案移到其他位置後不需修改路徑。

## Python 套件

所有直接使用的 Python 套件與測試版本列在 `requirements.txt`，主要包括：

- PyTorch：DQN 模型與訓練
- NumPy、SciPy：數值計算與 OCBA
- Gymnasium、PettingZoo、sumo-rl：強化學習環境介面
- SUMO `traci`、`sumolib`：控制 SUMO 模擬器
- Matplotlib：訓練結果繪圖
- Pandas：環境輸出資料處理

若要使用 NVIDIA GPU，請依
[PyTorch 官方安裝頁面](https://pytorch.org/get-started/locally/)選擇符合顯示卡驅動的 CUDA 版本。
