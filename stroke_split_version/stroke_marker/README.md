# Stroke Marker

## Introduction：
The labeling system is written in the Vite+Vue frameworks. This system is used for data labeling in the stroke segmentation font design process. For a new font design, hundreds of characters need to be labeled with vector strokes based on the designer's completed design, in order to be used for training neural networks and for better stroke segmentation network inference in stroke segmentation.

## QuickStart

```
npm install
npm run serve
```

## Usage：
This is a purely frontend labeling project. A JSON file processed from a font file needs to be uploaded, and the system will automatically parse it and form a labeling interface based on it. Users need to follow the guidelines for labeling. After each character is labeled, the browser will automatically download a 'character.json' file. After all characters have been labeled, all generated JSON files need to be packaged into a ZIP compressed package and left for upload on the frontend, with further instructions in the frontend file.
Below is a link to an example JSON file for input. The 'IDS' and 'IDS_EXPRESSION' fields for each font can be omitted:
 https://zjueducn-my.sharepoint.com/:u:/g/personal/21921307_zju_edu_cn/Ea93ClaBsYZIgzJ9zhYjgbQBMBhE3JSEZe7b6N-VVeftxQ?e=I01FPi
 (alternative link：https://gofile.me/73DMg/5162DQXo8 )

## Explanation on how to generate the uploaded JSON sample file

Please use the scripts in the stroke_marker_python_helper folder for generation. By executing the get_json_from_ttf.py script, a JSON file of characters from the char.txt file in a TTF font file can be created. 
- By modifying the file address on line 34, the input char file address can be changed. 
- By modifying the file address on line 76, the font file address can be changed. 
- This script will return the missing characters and the path of the JSON file.
