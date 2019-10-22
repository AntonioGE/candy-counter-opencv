#include <stdio.h>
#include <string>
#include <chrono>
#include <ctime>
#include <opencv2/opencv.hpp>
#include "opencv2/highgui.hpp"
#include "opencv2/highgui/highgui.hpp"

//Number of types of candies
const int NUM_TYPES = 15;

//IDs for shapes
const int BEAR = 0;
const int CIRCLE = 1;
const int SNAKE = 2;

//Number of colors and IDs
#define NUM_COLORS 7
const int BRIGHT_RED = 0;
const int DARK_RED = 1;
const int ORANGE = 2;
const int GREEN = 3;
const int YELLOW = 4;
const int BLACK_RED = 5;
const int WHITE = 6;

//Hue ranges for each color
std::vector<std::vector<std::vector<int>>> hueRanges = {
	{{0, 10}, {176, 179}},
	{{171, 179}},
	{{8, 20}},
	{{31, 53}},
	{{20, 34}},
	{{0, 179}},
	{{18, 32}}
};

//Saturation ranges for each color
std::vector<std::vector<std::vector<int>>> satRanges{
	{{130, 230}},
	{{130, 230}},
	{{120, 240}},
	{{85, 250}},
	{{120, 255}},
	{{0, 170}},
	{{16, 120}}
};

//Value ranges for each color
std::vector<std::vector<int>> valRanges{
	{30, 255},
	{30, 255},
	{30, 255},
	{30, 255},
	{30, 255},
	{0, 70},
	{30, 255}
};

const std::string colorNames[]{
	"BRIGHT_RED",
	"DARK_RED",
	"ORANGE",
	"GREEN",
	"YELLOW",
	"BLACK_RED",
	"WHITE"
};

std::vector<cv::Scalar> colorsScalar{
	cv::Scalar(58, 50, 160),
	cv::Scalar(22, 15, 106),
	cv::Scalar(46, 128, 203),
	cv::Scalar(96, 198, 21),
	cv::Scalar(63, 191, 200),
	cv::Scalar(0, 0, 0),
	cv::Scalar(255, 255, 255),
};

/*
std::vector<std::vector<int>> exampleResults{
	{1, 1, 1, 1, 1, 1,
	1, 1, 1, 1, 0, 1,
	1, 1, 1},

	{2, 2, 2, 2, 2, 2,
	2, 2, 2, 2, 2, 2,
	2, 2, 2},

	{3, 3, 3, 3, 3, 3,
	3, 3, 3, 3, 3, 3,
	3, 3, 3},

	{1, 0, 0, 1, 1, 0,
	2, 2, 2, 0, 1, 1,
	0, 1, 1},

	{4, 4, 2, 7, 5, 5,
	2, 2, 2, 0, 1, 1,
	0, 1, 1},

	{4, 4, 2, 7, 5, 5,
	2, 2, 2, 0, 1, 1,
	3, 3, 3},

	{4, 4, 2, 7, 5, 5,
	2, 2, 2, 0, 1, 1,
	3, 3, 3},

	{5, 5, 6, 4, 4, 5,
	2, 1, 3, 3, 2, 2,
	0, 0, 0},

	{5, 5, 6, 4, 4, 5,
	2, 3, 3, 5, 3, 5,
	1, 0, 0},

	{6, 7, 6, 9, 8, 8,
	3, 3, 3, 5, 3, 5,
	3, 3, 3},

	{2, 1, 0, 4, 3, 2,
	3, 2, 1, 1, 1, 1,
	1, 4, 0},

	{2, 3, 4, 4, 3, 3,
	6, 3, 2, 1, 1, 2,
	2, 5, 3},

	{2, 3, 4, 4, 3, 3,
	6, 3, 2, 1, 1, 2,
	2, 5, 3},

	{2, 3, 4, 4, 3, 8,
	6, 3, 2, 1, 2, 6,
	4, 5, 3},

	{2, 3, 4, 2, 3, 6,
	5, 3, 3, 3, 2, 4,
	4, 3, 2}
};
*/

int candyCount[NUM_TYPES];
int snakeSegmentsCount[NUM_COLORS];

const int numImages = 3;

cv::Mat src[numImages];
cv::Mat hsv[3];

int imageIndex = 0;

int cannyThresholdWhite = 23;
int cannyThresholdColored = 40;
float ratioThreshold = 0.4;
int boundsAreaLowThreshold = 700;

float maxCircleDiscrepancy = 0.19;//0.17 default?
float minDarkRedRatio = 0.65;//0.7 default?


//Get candy index by its shape and color (only for bear and circle)
int getCandyType(int shape, int color) {
	int type;
	if (shape == BEAR) {
		if (color < BLACK_RED) {
			type = color;
		}
		else {
			type = color - 1;
		}
	}
	else /* (shape == CIRCLE)*/ {
		if (color < BLACK_RED) {
			type = color + 6;
		}
		else {
			type = color + 5;
		}
	}
	return type;
}

//Resize image to maxSize keeping aspect ratio
void resizeKeepAspectRatio(cv::Mat &src, cv::Mat &dst, int maxSize) {
	if (src.size().width > src.size().height) {
		cv::resize(src, dst, cv::Size(maxSize, (src.size().height*maxSize) / src.size().width));
	}
	else {
		cv::resize(src, dst, cv::Size((src.size().width*maxSize) / src.size().height, maxSize));
	}
}

//Apply canny threshold to image
void CannyThreshold(cv::Mat src, cv::Mat &dst, int threshold) {
	int ratio = 4;
	const int kernel_size = 3;
	cv::blur(src, dst, cv::Size(kernel_size, kernel_size));
	cv::Canny(src, dst, threshold, threshold*ratio, kernel_size);
}

//Aplly morphological operation to image
void morphOperation(cv::Mat &src, int op, int value) {
	cv::morphologyEx(src, src, op, cv::getStructuringElement(2, cv::Size(value * 2 + 1, value * 2 + 1), cv::Point(value, value)));
}

//Count all the candies
void countCandy() {

	/*
		---FIND BLOBS USING HSV RANGES---
	*/

	//Split HSV channels into 3 matrices
	cv::Mat temp;
	cv::cvtColor(src[imageIndex], temp, cv::COLOR_BGR2HSV);
	cv::split(temp, hsv);

	//Create matrix array for storing blobs of each type of color
	cv::Mat colors[NUM_COLORS];

	//Find blobs using HSV ranges for each color type
	for (int i = 0; i < NUM_COLORS; i++) {

		//Find Hue blobs in range and mix them
		cv::Mat hueMix = cv::Mat::zeros(hsv[0].size(), CV_8UC1);
		for (int j = 0; j < hueRanges.at(i).size(); j++) {
			cv::Mat hueImage;
			cv::inRange(hsv[0], hueRanges[i][j][0], hueRanges[i][j][1],	hueImage);
			cv::bitwise_or(hueMix, hueImage, hueMix);
		}

		//Find Saturation blobs in range and mix them
		cv::Mat satMix = cv::Mat::zeros(hsv[1].size(), CV_8UC1);
		for (int j = 0; j < satRanges.at(i).size(); j++) {
			cv::Mat satImage;
			cv::inRange(hsv[1], satRanges[i][j][0], satRanges[i][j][1],	satImage);
			cv::bitwise_or(satMix, satImage, satMix);
		}

		//Find Value blobs in range
		cv::Mat valMix = cv::Mat::zeros(hsv[1].size(), CV_8UC1);
		cv::inRange(hsv[2], valRanges[i][0], valRanges[i][1], valMix);

		//Get the blobs that satisfy all the HSV ranges
		cv::bitwise_and(hueMix, satMix, colors[i]);
		cv::bitwise_and(colors[i], valMix, colors[i]);

		//GREEN color can produce YELLOW reflections, so we remove the GREEN blobs that intersect with YELLOW ones 
		if (i == YELLOW) {
			colors[YELLOW] -= colors[GREEN];
		}

		//Morph CLOSE operation for joining nearby blobs and open edges
		morphOperation(colors[i], cv::MORPH_CLOSE, 1);

		//Remove small blobs by moprh OPEN and CLOSE (RED and WHITE colors need high deteal so they are threated in a different manner)
		if (i != BRIGHT_RED && i != DARK_RED && i != WHITE) {
			morphOperation(colors[i], cv::MORPH_OPEN, 2);
			morphOperation(colors[i], cv::MORPH_CLOSE, 3);
		}

		//cv::imshow(colorNames[i], colors[i]);
	}

	/*
		---REFINE BLOBS USING EDGE DETECTION---
	*/

	//Apply a mask based on edges detected by Canny detector. Pixels inside closed edge will remain after this operation
	cv::Mat edgeMask;
	CannyThreshold(hsv[1], edgeMask, cannyThresholdColored);
	morphOperation(edgeMask, cv::MORPH_CLOSE, 2);
	//cv::imshow("Edges BEFORE", edgeMask);
	cv::floodFill(edgeMask, cv::Point(0, 0), 100);
	cv::inRange(edgeMask, 100, 100, edgeMask);
	cv::bitwise_not(edgeMask, edgeMask);
	morphOperation(edgeMask, cv::MORPH_OPEN, 2);
	//cv::imshow("Edges AFTER", edgeMask);

	//Create mask joining all the blobs of each colors except WHITE. Apply edge based mask to all colors except WHITE. 
	cv::Mat mask = cv::Mat::zeros(src[imageIndex].rows, src[imageIndex].cols, CV_8UC1);
	for (int i = 0; i < NUM_COLORS - 1; i++) {
		cv::bitwise_and(colors[i], edgeMask, colors[i]);
		cv::bitwise_or(colors[i], mask, mask);
	}
	//Dilate the mask for better performance
	int dilateValue = 1;
	cv::dilate(mask, mask, cv::getStructuringElement(0, cv::Size(dilateValue * 2 + 1, dilateValue * 2 + 1), cv::Point(dilateValue, dilateValue)));
	cv::bitwise_not(mask, mask);


	//cv::imshow("MASK FOR WHITE", mask);

	//Calculate the edges for white candy
	cv::Mat whiteEdges;
	CannyThreshold(src[imageIndex], whiteEdges, cannyThresholdWhite);
	//Apply the color mask to the white edge mask
	cv::bitwise_and(whiteEdges, mask, whiteEdges);
	//Close the edges
	morphOperation(whiteEdges, cv::MORPH_CLOSE, 4);
	//Fill the inner parts of the blobs. NOTE: this part could fail if candy are very close each other 
	cv::floodFill(whiteEdges, cv::Point(0, 0), 100);
	cv::inRange(whiteEdges, 100, 100, whiteEdges);
	cv::bitwise_not(whiteEdges, whiteEdges);
	//Remove small blobs with morph OPEN
	morphOperation(whiteEdges, cv::MORPH_OPEN, 4);

	//Store the white mask into the colors array
	colors[NUM_COLORS - 1] = whiteEdges;
	//cv::imshow("NEW WHITE", colors[WHITE]);

	/*
		---REFINE BRIGHT RED AND DARK RED BLOBS---
	*/

	//BrightRed and DarkRed candy can have blobs situated in the same position
	//This code joins intersecting blobs and determines its color (BrightRed or DarkRed)
	cv::Mat redMix;
	cv::bitwise_or(colors[BRIGHT_RED], colors[DARK_RED], redMix);
	//Remove also parts from black parts of snake (that also shares red tones)
	redMix -= colors[BLACK_RED];
	//cv::imshow("RED MIX", redMix);

	//Create empty images for storing new BrightRed and DarkRed blobs
	cv::Mat newBrightRed = cv::Mat::zeros(colors[BRIGHT_RED].size(), colors[BRIGHT_RED].type());
	cv::Mat newDarkRed = cv::Mat::zeros(colors[DARK_RED].size(), colors[DARK_RED].type());

	//Detect contours of each blob in the red mix
	std::vector<std::vector<cv::Point>> redContours;
	std::vector<cv::Vec4i> redHierarchy;
	cv::findContours(redMix, redContours, redHierarchy, CV_RETR_TREE, CV_CHAIN_APPROX_SIMPLE, cv::Point(0, 0));
	//Determine the red tone for each blob
	for (int i = 0; i < redHierarchy.size(); i++) {
		//Get bounds of the blob and calculate area
		cv::Rect bounds = cv::boundingRect(redContours[i]);
		int boundsArea = bounds.width * bounds.height;

		//Discard small blobs
		if (boundsArea > boundsAreaLowThreshold) {
			//Create image of the size of the blob and paint it
			cv::Mat mixImg = cv::Mat::zeros(bounds.height, bounds.width, CV_8UC1);
			cv::drawContours(mixImg, redContours, i, 255, CV_FILLED, 8,	std::vector<cv::Vec4i>(), 0, cv::Point(-bounds.x, -bounds.y));

			//Create images of size of the mix blob and copy the blobs from each tone
			cv::Mat brightRedImg = colors[BRIGHT_RED](bounds).clone();
			cv::Mat darkRedImg = colors[DARK_RED](bounds).clone();
			
			//Get only the parts inside the blob mix
			cv::bitwise_and(brightRedImg, mixImg, brightRedImg);
			cv::bitwise_and(darkRedImg, mixImg, darkRedImg);

			//Count how many pixels has the mix blob and the dark and bright blobs and calculate their ratios
			int mixNonZero = cv::countNonZero(mixImg);
			//float brightRedRatio = (float)(cv::countNonZero(brightRedImg)) / mixNonZero;//UNUSED
			float darkRedRatio = (float)(cv::countNonZero(darkRedImg)) / mixNonZero;

			//If the ratio of the dark blob is higher than the minimum stablished, the blob is dark. If not, it is bright
			if (darkRedRatio > minDarkRedRatio) {
				mixImg.copyTo(newDarkRed(bounds));
			}
			else {
				mixImg.copyTo(newBrightRed(bounds));
			}

		}
	}
	colors[BRIGHT_RED] = newBrightRed;
	colors[DARK_RED] = newDarkRed;
	
	//Remove small blobs and artifacts
	morphOperation(colors[BRIGHT_RED], cv::MORPH_CLOSE, 5);
	morphOperation(colors[BRIGHT_RED], cv::MORPH_OPEN, 5);
	morphOperation(colors[DARK_RED], cv::MORPH_CLOSE, 5);
	morphOperation(colors[DARK_RED], cv::MORPH_OPEN, 5);


	cv::Mat tempRedCheck;
	resizeKeepAspectRatio(colors[BRIGHT_RED], tempRedCheck, 700);
	//cv::imshow("BRIGHT_RED SMALL", tempRedCheck);
	//cv::imshow("BRIGHT_RED POST", colors[BRIGHT_RED]);
	//cv::imshow("DARK_RED POST", colors[DARK_RED]);

	cv::Mat imageProcessed = src[imageIndex].clone();

	/*
		---DETERMINE SHAPE AND COUNT CANDIES---
	*/

	//Reset candy counters
	for (int i = 0; i < NUM_TYPES; i++) {
		candyCount[i] = 0;
	}
	for (int i = 0; i < NUM_COLORS; i++) {
		snakeSegmentsCount[i] = 0;
	}

	//For each color matrix, find the blobs and determine its shape
	for (int i = 0; i < NUM_COLORS; i++) {
		std::vector<std::vector<cv::Point> > contours;
		std::vector<cv::Vec4i> hierarchy;
		cv::findContours(colors[i], contours, hierarchy, CV_RETR_TREE, CV_CHAIN_APPROX_SIMPLE, cv::Point(0, 0));
	
		//Determine the shape of each contour
		for (int j = 0; j < hierarchy.size(); j++) {
			//Get blob bounds and calculate area
			cv::Rect bounds = cv::boundingRect(contours[j]);
			int boundsArea = bounds.width * bounds.height;

			//Discard small blob
			if (boundsArea > boundsAreaLowThreshold) {
				//Draw blob in an image with the size of the bounds
				cv::Mat blobImage = cv::Mat::zeros(bounds.height, bounds.width, CV_8SC1);
				cv::drawContours(blobImage, contours, j, 127, CV_FILLED, 8,	std::vector<cv::Vec4i>(), 0, cv::Point(-bounds.x, -bounds.y));

				//Draw circle in an image with the size of the bounds
				cv::Mat circleImage = cv::Mat::zeros(bounds.height,	bounds.width, CV_8SC1);
				cv::circle(circleImage, cv::Point(bounds.width / 2,	bounds.height / 2), cv::min(bounds.width, bounds.height) / 2, 127, CV_FILLED, 8);

				//Compare the blob with the circle by counting how many pixels are different. Calculate its discrepancy
				int nonZero = cv::countNonZero(cv::abs(blobImage - circleImage));
				float circleDiscrepancy = (float)nonZero / boundsArea;
				//Calculate aspect ratio of the bounds
				float aspectRatio = (float)(std::max(bounds.height, bounds.width)) / std::min(bounds.height, bounds.width);
				//Calculate solidity of the blob
				float solidity = (float)(boundsArea) / cv::contourArea(contours[j]);

				//If the circleDiscrepancy is less than the maximum, then it is a circle
				if (circleDiscrepancy < maxCircleDiscrepancy) {
					//std::cout << "circle " << colorNames[i] << " " << bounds.x << " " << bounds.y << std::endl;
					cv::circle(imageProcessed, cv::Point(bounds.x +	bounds.width / 2, bounds.y + bounds.height / 2), cv::min(bounds.width, bounds.height) / 2, colorsScalar[i], 2, 8);
					cv::putText(imageProcessed, "circle", cv::Point(bounds.x, bounds.y), 0, 0.6, cv::Scalar(0, 0, 0), 1, 8, false);
					candyCount[getCandyType(1, i)]++;
				}
				//Else, check if the blob is a snake by checking the solidity, aspect ratio and area
				else if ((solidity > 2 || aspectRatio > 2.5 || (boundsArea > 3400 && aspectRatio > 2)) && (boundsArea > 2000)) {
					//std::cout << "snake " << colorNames[i] << " " << 	bounds.x << " " << bounds.y << std::endl;
					cv::rectangle(imageProcessed, bounds, colorsScalar[i], 2);
					cv::putText(imageProcessed, "snake", cv::Point(bounds.x, bounds.y), 0, 0.6, cv::Scalar(0, 0, 0), 1, 8, false);
					//candyCount[getCandyType(2, i)]++;
					snakeSegmentsCount[i]++;
				}
				//Else the shape is a bear
				else {
					//std::cout << "bear " << colorNames[i] << " " << 	bounds.x << " " << bounds.y << std::endl;
					cv::rectangle(imageProcessed, bounds, colorsScalar[i], 1);
					cv::putText(imageProcessed, "bear", cv::Point(bounds.x, bounds.y), 0, 0.6, cv::Scalar(0, 0, 0), 1, 8, false);
					candyCount[getCandyType(0, i)]++;
				}
			}
		}
	}

	//Count how many snakes we have
	candyCount[12] = snakeSegmentsCount[GREEN];
	candyCount[13] = snakeSegmentsCount[ORANGE];
	candyCount[14] = snakeSegmentsCount[YELLOW];
	//In the case WHITE-GREEN snake parts counting are different, then PROBABLY a white bear was threated as a snake. Then it is corrected here
	candyCount[5] += (snakeSegmentsCount[WHITE] - snakeSegmentsCount[GREEN]);

	std::cout << std::endl;
	for (int i = 0; i < NUM_TYPES; i++) {
		//std::cout << "Type " << i << " real: " << exampleResults[imageIndex][i] <<" counted: " << candyCount[i] << " diff: " << exampleResults[imageIndex][i] - candyCount[i]	<< std::endl;
	}

	//cv::imshow("Image processed", imageProcessed);
	resizeKeepAspectRatio(imageProcessed, imageProcessed, 750);
	cv::imshow("Image processed SMALL", imageProcessed);
	cv::Mat imageSmall;
	resizeKeepAspectRatio(src[imageIndex], imageSmall, 750);
	cv::imshow("image SMALL", imageSmall);
	//cv::imshow("image", src[imageIndex]);
}


void trackbarImageSelectorCallback(int, void*) {
	countCandy();
}

int main() {

	std::string inputPath = "TestImages";
	std::string outputPath = "out";
	std::string inputFileName = "img_";

	//Stablish maximum size of the biggest dimension of the image (width, height)
	int maxSize = 1000;

	//Load image array
	for (int i = 0; i < numImages; i++) {
		std::stringstream ss;
		ss << std::setw(3) << std::setfill('0') << i;
		std::string number = ss.str();
		src[i] = cv::imread(inputPath + "/" + inputFileName + number + ".jpg", CV_LOAD_IMAGE_COLOR);
		if (!src[i].data) {
			std::cout << "Can not open image";
			return 0;
		}

		resizeKeepAspectRatio(src[i], src[i], maxSize);
	}

	//Count candies
	countCandy();

	//Menu control
	cv::namedWindow("Menu", CV_WINDOW_AUTOSIZE);
	cv::createTrackbar("Image index", "Menu", &imageIndex, numImages - 1, trackbarImageSelectorCallback);


	cv::waitKey(0);

	return 0;
}


