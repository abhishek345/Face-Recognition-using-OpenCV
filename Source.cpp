#include <opencv2/highgui/highgui.hpp>
#include <opencv2/objdetect/objdetect.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/contrib/contrib.hpp>

#include<iostream>
#include<string>
#include<stdio.h>
#include<fstream>
#include<sstream>

using namespace std;
using namespace cv;

String face_cascade_name = "C:/OpenCV 2.4.13/opencv/sources/data/haarcascades/haarcascade_frontalface_alt2.xml";
CascadeClassifier face_cascade;

static void read_csv(const string& filename, vector<Mat>& images, vector<int>& labels, char separator = ';') {
	std::ifstream file(filename.c_str(), ifstream::in);
	if (!file) {
		string error_message = "No valid input file was given, please check the given filename.";
		CV_Error(CV_StsBadArg, error_message);
	}
	string line, path, classlabel;
	Mat temp;

	while (getline(file, line)) {
		stringstream liness(line);
		getline(liness, path, separator);
		getline(liness, classlabel);
		if (!path.empty() && !classlabel.empty()) {
			temp = imread(path, 0);
			cvtColor(temp, temp, CV_BayerRG2GRAY);
			images.push_back(temp);
			labels.push_back(atoi(classlabel.c_str()));
		}
	}
}

int main(int argc, char** argv)
{
	// Get the path to your CSV:
	string fn_csv = "C:/Users/sasi.smart/Documents/0_UT Dallas/2nd Sem/VA/Assignment 4/Assignment 4_Abhishek/train.csv";
	// These vectors hold the images and corresponding labels
	vector<Mat> images;
	vector<int> labels;

	// Read in the data 
	try {
		read_csv(fn_csv, images, labels);
	}
	catch (cv::Exception& e) {
		cerr << "Error opening file \"" << fn_csv << "\". Reason: " << e.msg << endl;
		exit(1);
	}

	int im_width = images[0].cols;
	int im_height = images[0].rows;

	for (int i = 1; i < images.size(); i++){
		if (im_height > images[i].rows && im_width > images[i].cols){
			im_width = images[i].cols;
			im_height = images[i].rows;
		}
	}

	// Create a FaceRecognizer and train it on the given images:
	Ptr<FaceRecognizer> model = createFisherFaceRecognizer();
	model->train(images, labels);

	if (!face_cascade.load(face_cascade_name))
	{
		printf("Error loading face cascade");
		return -1;
	}

	VideoCapture cap(0);
	if (!cap.isOpened())
	{
		return -1;
	}


	Mat edges;
	for (;;)
	{
		Mat frame;
		cap >> frame;
		Mat original = frame.clone();
		if (!frame.empty())
		{

			// Convert the current frame to grayscale:
			Mat gray;
			cvtColor(original, gray, CV_BGR2GRAY);
			// Find the faces in the frame:
			vector< Rect_<int> > faces;
			face_cascade.detectMultiScale(gray, faces);
			for (int i = 0; i < faces.size(); i++) {
				Rect face_i = faces[i];
				// Crop the face from the image
				Mat face = gray(face_i);
				
				Mat face_resized;
				cv::resize(face, face_resized, Size(im_width, im_height), 1.0, 1.0, INTER_CUBIC);
				// Perform the prediction
				int prediction = model->predict(face_resized);
				
				// Draw a green rectangle around the detected face:
				rectangle(original, face_i, CV_RGB(0, 255, 0), 1);
				
				int pos_x = std::max(face_i.tl().x - 10, 0);
				int pos_y = std::max(face_i.tl().y - 10, 0);
				
				if (prediction == 0)
					GaussianBlur(original(faces[i]), original(faces[i]), Size(0,0), 20);
			}

		}
		else
		{
			printf("No frame captured");
			break;
		}
		imshow("face recognizer", original);

		if ((char)waitKey(1) == (char)27)
			break;

	}
		return 0;
	
}

