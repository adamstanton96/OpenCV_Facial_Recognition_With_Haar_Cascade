/*=============================================================================
* Project: AI Assignment 2 - Facial Recognition.
* Adam Stanton, B00266256.
=============================================================================*/

#include "opencv2/core.hpp"
#include "opencv2/face.hpp"
#include "opencv2/objdetect.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"

#include <iostream>
#include <fstream>
#ifdef __GNUC__
#include <experimental/filesystem> // Full support in C++17
namespace fs = std::experimental::filesystem;
#else
#include <filesystem>
namespace fs = std::tr2::sys;
#endif

// https://msdn.microsoft.com/en-us/library/dn986850.aspx
// GCC 7.2.0 Ok on Linux
// g++ -std=c++1z 1_simple_facerec_eigenfaces.cpp -lopencv_face -lopencv_core -lopencv_imgcodecs -lstdc++fs

////////////////////////////////////////////////////////////////////////
//Resizes an image but keeps aspect ratio - use in place of cv::Resize//
////////////////////////////////////////////////////////////////////////
cv::Mat fixedAspectResize(const cv::Mat& inputSample, int targetWidth, int targetHeight)
{
	int width = inputSample.cols;
	int height = inputSample.rows;

	cv::Mat fixedSample = cv::Mat::zeros(targetHeight, targetWidth, inputSample.type());

	int maxDim = (width >= height) ? width : height;
	float scale = ((float)targetWidth) / maxDim;
	
	cv::Rect roi;	//Region Of Interest (ROI) i.e. the region of the mat containing the face.
	if (width >= height)
	{
		roi.width = targetWidth;
		roi.x = 0;
		roi.height = height * scale;
		roi.y = (targetHeight - roi.height) / 2;
	}
	else
	{
		roi.y = 0;
		roi.height = targetHeight;
		roi.width = width * scale;
		roi.x = (targetWidth - roi.width) / 2;
	}

	//Use cv::resize to resize the sample...
	cv::resize(inputSample, fixedSample(roi), roi.size());
	//Return the resized, aspect-fixed sample...
	return fixedSample;
}

/////////////////////////////////////////////////////////////////////////////
//Uses a haar cascade to find and a face contained within the parameter Mat//
/////////////////////////////////////////////////////////////////////////////
cv::Mat cascadeFaceDetection(cv::Mat frame, cv::CascadeClassifier faceCascade, cv::CascadeClassifier eyeCascade)
{
	std::vector<cv::Rect> faces;
	cv::Mat sample;		//Used to store greyscale representation of our input.
	cv::Mat faceROI;	//Region Of Interest (ROI) i.e. the region of the mat containing the face.
	const char windowName[] = "Face_Detection";

	//Ensure the image is greyscale, required for face recognition...
	cvtColor(frame, sample, cv::COLOR_BGR2GRAY);
	equalizeHist(sample, sample);

	//Use the cascade to find the face/s in the sample.
	faceCascade.detectMultiScale(sample, faces, 1.1, 2, 0 | cv::CASCADE_SCALE_IMAGE, cv::Size(30, 30));

	for (size_t i = 0; i < faces.size(); i++)
	{
		//Find the center of the face...
		cv::Point center(faces[i].x + faces[i].width*0.5, faces[i].y + faces[i].height*0.5);
		//Draw a circle around it on the input frame (will be seen when frame is drawn)...
		ellipse(frame, center, cv::Size(faces[i].width*0.5, faces[i].height*0.5), 0, 0, 360, cv::Scalar(255, 0, 255), 4, 8, 0);

		faceROI = sample(faces[i]);
		std::vector<cv::Rect> eyes;

		//Find the eyes within the face...
		eyeCascade.detectMultiScale(faceROI, eyes, 1.1, 2, 0 | cv::CASCADE_SCALE_IMAGE, cv::Size(30, 30));

		for (size_t j = 0; j < eyes.size(); j++)
		{
			//Find the center of the eye...
			cv::Point center(faces[i].x + eyes[j].x + eyes[j].width*0.5, faces[i].y + eyes[j].y + eyes[j].height*0.5);
			int radius = cvRound((eyes[j].width + eyes[j].height)*0.25);
			//Draw a circle around it on the input frame (will be seen when frame is drawn)...
			circle(frame, center, radius, cv::Scalar(255, 0, 0), 4, 8, 0);
		}
	}
	//Display the frame (circling detected faces)...
	imshow(windowName, frame);
	//Return the face...
	return faceROI;
}

int main(int argc, char *argv[])
{
	//Used to hold the images from the att_faces database...
	std::vector<cv::Mat> images;
	std::vector<int>     labels;

	//Populate the vectors with the faces...
	//Iterate through all subdirectories, looking for .pgm files...
	fs::path p(argc > 1 ? argv[1] : "../att_faces");
	for (const auto &entry : fs::recursive_directory_iterator{ p })
	{
		if (entry.path().extension() == ".pgm") 
		{
			std::string str = entry.path().parent_path().stem().string(); // s26 s27 etc.
			int label = atoi(str.c_str() + 1); // s1 -> 1
			//Push the image and label back into the vectors...
			images.push_back(cv::imread(entry.path().string().c_str(), 0));
			labels.push_back(label);
		}
	}

	//Perform training...
	std::cout << "Training in progress..." << std::endl;
	cv::Ptr<cv::face::BasicFaceRecognizer> model = cv::face::createEigenFaceRecognizer();
	model->train(images, labels);
	std::cout << "Training complete." << std::endl;

	//Set up the haar cascades for face recognition...
	cv::CascadeClassifier faceCascade;
	cv::CascadeClassifier eyeCascade;

	if (!faceCascade.load("../assets/haarcascade_frontalface_alt.xml"))
	{
		std::cout << "error: face cascade could not be loaded." << std::endl;
		return -1;
	}

	if (!eyeCascade.load("../assets/haarcascade_eye_tree_eyeglasses.xml"))
	{
		std::cout << "error: eye cascade could not be loaded." << std::endl;
		return -1;
	}

					
	cv::Mat frame;	//The current frame of camera input.
	cv::Mat sample;	//The sample we will perform our predictions on.

	//Initialize our video input stream, argument is the camera id...
	cv::VideoCapture vid_in(0);  
	//Check we have opened the camera, if not present error and exit...
	if (!vid_in.isOpened()) 
	{
		std::cout << "error: Camera 0 could not be opened for capture." << std::endl;
		return -1;
	}

	const char windowName[] = "FaceForComparison";
	cv::namedWindow(windowName);
	while (1)
	{
		//Update the frame with the current camera input...
		vid_in >> frame;

		//If camera input has been logged...
		if(!frame.empty())
			//Update the sample mat with a face returned by our cascade...
			sample = cascadeFaceDetection(frame, faceCascade, eyeCascade);

		//If our cascade has returned a face...
		if (!sample.empty())
		{
			//Resize it to match our other faces (92 x 112, Greyscale already applied by cascade.)...
			sample = fixedAspectResize(sample, 92, 112);

			int predictedLabel;	//Index of the face predicted by our predict method.
			double confidence;	//The number of standard deviations away from the mean.
						
			//Make a prediction based on the sample face...
			model->predict(sample, predictedLabel, confidence);
			std::cout << "\nPredicted Class: " << predictedLabel << " - Confidence: " << confidence << '\n';	//A lower confidence value means the prediciton is -believed to be- more accurate.
		}
		else
		{
			//If our casacde has not found a face, load an advisory image to be displayed in its absence...
			sample = cv::imread("../assets/noFacesFound.pgm", cv::IMREAD_GRAYSCALE);
		}
		
		//Show the sample face (or advisory image) in another window...
		imshow(windowName, sample);

		if (cv::waitKeyEx(1000 / 30) >= 0) // how long to wait for a key (milliseconds) (30 defines FPS).
			break;
	}

	//Close the video input stream before ending the program...
	vid_in.release();
	return 0;
}