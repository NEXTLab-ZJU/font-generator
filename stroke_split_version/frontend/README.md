# Stroke Split Version FrontEnd
The front end is written in the Vue framework and communicates with the back end using the Axios library. The Canvas is used for drawing related content. Before running the front-end project, please start the back-end algorithmic service and complete the labeling task of the stroke_marker system first.

## QuickStart
1. install requirements
```
npm install
```
2. modify config
Open src/config.js，modify backend request url

3. run
```
npm run serve
```

## Usage：
1. The system requires the upload of a ZIP compressed package containing a font bitmap already designed by the designer. The ZIP package must contain font bitmap images. A sample ZIP compressed package can be downloaded from the following link:
 https://zjueducn-my.sharepoint.com/:u:/g/personal/21921307_zju_edu_cn/EQvRsZFXm9NFl4xfLkOmrlwBZwdcxdnpXz9iwrZvFnl8Ig?e=NHvbFh
 (alternative link：https://gofile.me/73DMg/WV62YGV1m)
2. In addition, a ZIP compressed package containing the JSON file that has been labeled by the stroke_marker system needs to be uploaded. The ZIP package must contain JSON files. A sample ZIP package can be downloaded from the following link. For more information on how to obtain the JSON file, please refer to the stroke_marker system instructions:
  https://zjueducn-my.sharepoint.com/:u:/g/personal/21921307_zju_edu_cn/EXqOn7iDZ11PvD3tO6laEPEBIQJJPDxB3gdVNoK7DNr6uw?e=8upSKN 
(alternative link https://gofile.me/73DMg/FzpZJmdQm )
3. Afterwards, the system will automatically start training. Normally, training takes about 2-3 days to complete. After training is completed, relevant vectorization operations can be performed within the system. You can also directly access the corresponding folder in the backend to view the results.
