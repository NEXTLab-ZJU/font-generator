# Font Generator
## Background：
The design of Chinese fonts can be divided into four stages: proposal, production, testing and optimization, and release. The proposal stage is the stage of determining the font style, such as glyph design, weight design, and kerning design, which focuses on determining the shape, line, and proportion of the font to ensure the visual balance and readability of the entire font. The production stage is usually a labor-intensive work, where designers, according to the font style determined in the proposal stage, produce a large character set such as GB2312 in batches. During the testing and optimization stage, designers mainly use testing and optimization tools such as glyph alignment, kerning evenness, and letter shape consistency to check and optimize font design. Font designers first need to test the design of each glyph to ensure that they look clear and accurate at different sizes and resolutions. After testing the glyph design, font designers need to check and optimize the font outline to ensure that the font has the best appearance in various sizes and environments. This process usually involves adjusting the curve, stroke width, kerning, glyph ratio, etc. Font designers need to do comprehensive testing and fix any errors or issues that may occur, such as broken strokes or strange shapes. Finally, font designers need to conduct rendering tests to ensure that the font can be displayed correctly on different operating systems, software, and devices without any visual problems.
The number of characters to design in the proposal stage is usually a few hundred to a thousand, while the number of characters to design in the production stage is tens or hundreds of times that of the proposal stage. In the proposal, testing and optimization, and release stages, the design of the proposal, the control of details and optimization is the time that designers must spend, which has a high degree of freedom and can truly reflect the design quality of senior font designers. In the mass production stage, more work is repetitive labor according to established standards, which belongs to labor-intensive work.


## System Target
This system focuses on the production stage of font design, aiming to help font designers improve their design and production efficiency in the large character set production stage based on existing font design schemes. The system enables designers to quickly complete the design of different glyphs at this stage.

## System Design Principles
### 1、User-friendly：
-  This font design tool is intuitive, easy to understand, and use. Designers should be able to get started quickly and use the tool without additional training.
### 2、Improved design efficiency：
- This auxiliary tool focuses on the production phase, simplifying the vector outlining process for each character in the production phase by combining existing open-source large-scale model-based image generation techniques with the proprietary AI vectorization technology developed in this project, thereby improving design efficiency.
### 3、Adapt to different design scenarios：
- There are often two types of design scenarios in font design. One is the design of standard fonts, which often requires considering multiple extended features, such as increasing variable font features and multiple weight characteristics. Therefore, it is necessary to split the vectors of different strokes, such as commonly used Kai, Song, and Hei fonts. The other is the design of artistic fonts, which often does not require variable font features and supports fewer features. The vector strokes do not need to be split in the design process. The system proposes two sets of design processes that can adapt to different design scenarios.

## Project Intro：
We have designed and completed two sets of complete front-end and back-end systems, combined with artificial intelligence algorithms, for two scenarios:

- Non-standard font (artistic font) design scenario:

Inside the 'non_stroke_split_version' folder, there are backend and frontend components. For specific startup instructions, please refer to the corresponding internal readme.md file.
Such fonts often do not require variable font features and have limited support. During the design process, vector strokes do not need to be split.

- Standard font (Song, Kai, Hei) design scenario:

Inside the 'stroke_split_version' folder, you will find backend and frontend components. For specific startup instructions, please refer to the corresponding internal readme.md file. Additionally, there is a 'stroke_marker' folder, which contains the frontend system for labeling and can be independently run and built for labeling purposes.
These types of fonts often require consideration of multiple extended features, such as variable font features, multiple weights, etc. Because of these features, it is necessary to split the vectors of different strokes, as seen in common fonts like Kai, Song, Hei, and others