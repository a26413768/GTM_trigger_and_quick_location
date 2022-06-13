# Requirements
- pandas
- numpy
- scipy
- matplotlib
- astropy
- pyquaternion
- skyfield
- sklearn

# Usage
## Enlarge simulated respones table
把simulated respones table放進table資料夾底下，將enlarge_table.py裡的變數file_name的值改成跟simulated respones table一致。

run 
```bash
python enlarge_table.py
```
執行完，經內插變大的table會加上big_在原檔名前，並一樣放在table資料夾下方。
## Trigger
將待測"level 1檔案"和"衛星姿態檔案"放在input資料夾裡面，將trigger.py裡的變數filename的值改成跟"level 1檔案" 一致、filename_sc改成跟"衛星姿態檔案"一致。

run 
```bash
python trigger.py
```
如果有trigger事件，會產生三張圖片、一個CSV檔在output資料夾裡，
1. 以最小trigger bin binning的light curve
2. 以0.1秒binning的light cruve
3. 以最小trigger bin binning的cumulative light curve
4. 包含trigger 資訊的CSV檔

如果沒有trigger事件，只會產生兩張圖片
1. 以2秒binning的light curve
2. 以0.1秒binning的light cruve

## location
需要先確保有三個enlarged table放在table資料夾裡，將trigger產生的CSV檔放在output資料夾。

run 
```bash
python localization.py
```
執行完會產生兩張圖、一個CSV檔在location_output資料夾裡
1. 全天圖，包含localization位置、地球、太陽、銀河等
2. 部分天圖，包含best fit位置、1-2-3 sigma範圍
3. localization的結果與相關資訊