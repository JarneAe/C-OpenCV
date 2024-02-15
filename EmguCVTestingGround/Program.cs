using System;
using System.Collections.Generic;
using System.Drawing;
using AForge.Video.DirectShow;
using Emgu.CV;
using Emgu.CV.CvEnum;
using Emgu.CV.Structure;
using Emgu.CV.Util;

namespace OpenCVTesting
{
    class Program
    {
         static void Main(string[] args)
        {
                       // Get a list of all video devices
            Console.WriteLine("Choose an option:");
            Console.WriteLine("1. Use an image");
            Console.WriteLine("2. Use a webcam");
            Console.Write("Enter your choice (1 or 2): ");
            string choice = Console.ReadLine();

            if (choice == "1")
            {
                ProcessImage();
            }
            else if (choice == "2")
            {
                ProcessWebcam();
            }
            else
            {
                Console.WriteLine("Invalid choice. Exiting...");
            }
        }

        static void ProcessImage()
        {
            Console.Write("Enter the path to the image file: ");
            string imagePath = Console.ReadLine();

            if (!System.IO.File.Exists(imagePath))
            {
                Console.WriteLine("Image file not found.");
                return;
            }

            Mat image = CvInvoke.Imread(imagePath, ImreadModes.Color);
            if (image.IsEmpty)
            {
                Console.WriteLine("Unable to read the image file.");
                return;
            }

            Dictionary<string, Bgr> colorsToCheck = new Dictionary<string, Bgr>()
            {
                { "Red", new Bgr(0, 0, 255) },
                { "Green", new Bgr(0, 255, 0) },
                { "Blue", new Bgr(255, 0, 0) }
            };

            CheckForColorsInGrid(image, colorsToCheck);
            CvInvoke.Imshow("Image", image);
            CvInvoke.WaitKey(0);
            CvInvoke.DestroyAllWindows();
        }

        static void ProcessWebcam()
        {
            FilterInfoCollection videoDevices = new FilterInfoCollection(FilterCategory.VideoInputDevice);

            if (videoDevices.Count == 0)
            {
                Console.WriteLine("No webcams found.");
                return;
            }
            
            VideoCapture capture = new VideoCapture(0);

            if (!capture.IsOpened)
            {
                Console.WriteLine("Unable to access webcam.");
                return;
            }

            Console.WriteLine("Webcam found: " + videoDevices[0].Name);

            Mat image = new Mat();

       
            while (true)
            {
             
                capture.Read(image);
                
                if (image.IsEmpty)
                {
                    Console.WriteLine("Unable to capture frame.");
                    break;
                }

                
                Dictionary<string, Bgr> colorsToCheck = new Dictionary<string, Bgr>()
                {
                    { "Red", new Bgr(0, 0, 255) },
                    { "Green", new Bgr(0, 255, 0) },
                    { "Blue", new Bgr(255, 0, 0) }
                };

                CheckForColorsInGrid(image, colorsToCheck);
                
                CvInvoke.Imshow("Captured Image", image);
                
                if (CvInvoke.WaitKey(1) == 27) // Esc key
                    break;
            }

     
            capture.Dispose();
            CvInvoke.DestroyAllWindows();
        }
        
        static void CheckForColorsInGrid(Mat image, Dictionary<string, Bgr> colorsToCheck)
        {
            Mat grayImage = new Mat();
            CvInvoke.CvtColor(image, grayImage, ColorConversion.Bgr2Gray);

            Mat cannyEdges = new Mat();
            CvInvoke.Canny(grayImage, cannyEdges, 50, 150);

            VectorOfVectorOfPoint contours = new VectorOfVectorOfPoint();
            Mat hierarchy = new Mat();
            CvInvoke.FindContours(cannyEdges, contours, hierarchy, RetrType.Tree, ChainApproxMethod.ChainApproxSimple);

            for (int i = 0; i < contours.Size; i++)
            {
                if (CvInvoke.ContourArea(contours[i]) > 10000)
                {
                    CvInvoke.DrawContours(image, contours, i, new MCvScalar(0, 0, 255), 2);
                    break;
                }
            }

            List<VectorOfPoint> innerSlots = new List<VectorOfPoint>();
            List<int> usedIndices = new List<int>();

            for (int i = 0; i < contours.Size; i++)
            {
                if (CvInvoke.ContourArea(contours[i]) < 10000)
                {
                    if (!usedIndices.Contains(i))
                    {
                        innerSlots.Add(contours[i]);
                        usedIndices.Add(i);

                        var moments = CvInvoke.Moments(contours[i]);
                        int centroidX = (int)(moments.M10 / moments.M00);
                        int centroidY = (int)(moments.M01 / moments.M00);


                        for (int j = i + 1; j < contours.Size; j++)
                        {
                            if (!usedIndices.Contains(j))
                            {
                                var moments2 = CvInvoke.Moments(contours[j]);
                                int centroidX2 = (int)(moments2.M10 / moments2.M00);
                                int centroidY2 = (int)(moments2.M01 / moments2.M00);

                                double distance = Math.Sqrt(Math.Pow(centroidX2 - centroidX, 2) +
                                                            Math.Pow(centroidY2 - centroidY, 2));

                                if (distance < 20)
                                {
                                    usedIndices.Add(j);
                                }
                            }
                        }
                    }
                }
            }

            innerSlots.Reverse();

            if (innerSlots.Count == 9)
            {
                for (int i = 0; i < innerSlots.Count; i++)
                {
                    VectorOfVectorOfPoint slotContour = new VectorOfVectorOfPoint();
                    slotContour.Push(innerSlots[i]);

                    //CvInvoke.DrawContours(image, slotContour, -1, new MCvScalar(0, 255, 0), 2);

                    var moments = CvInvoke.Moments(innerSlots[i]);
                    int centroidX = (int)(moments.M10 / moments.M00);
                    int centroidY = (int)(moments.M01 / moments.M00);

                    CvInvoke.PutText(image, (i + 1).ToString(), new Point(centroidX, centroidY), FontFace.HersheyComplex, 1, new MCvScalar(0, 0, 0), 1);

                    
                    foreach (var colorPair in colorsToCheck)
                    {
                        if (CheckForColor(image, innerSlots[i], colorPair.Value))
                        {
                            Console.WriteLine($"Grid {i + 1} contains {colorPair.Key.ToLower()} color.");
                        }
                    }
                }

                Console.WriteLine("Right amount of grid slots detected! proceed...");
            }

            CvInvoke.Imshow("Detected Grid", image);
            CvInvoke.WaitKey(0);
        }

        static bool CheckForColor(Mat image, VectorOfPoint contour, Bgr colorToCheck)
        {
            Rectangle rect = CvInvoke.BoundingRectangle(contour);
            using (Mat croppedImage = new Mat(image, rect))
            {
                Image<Bgr, byte> bgrImage = croppedImage.ToImage<Bgr, byte>();
                for (int x = 0; x < bgrImage.Width; x++)
                {
                    for (int y = 0; y < bgrImage.Height; y++)
                    {
                        Bgr pixel = bgrImage[y, x];
                        if (AreColorsSimilar(pixel, colorToCheck))
                        {
                            return true;
                        }
                    }
                }
            }
            return false;
        }

        static bool AreColorsSimilar(Bgr color1, Bgr color2)
        {
            int threshold = 50; 
            double distance = Math.Sqrt(Math.Pow(color1.Red - color2.Red, 2) +
                                        Math.Pow(color1.Green - color2.Green, 2) +
                                        Math.Pow(color1.Blue - color2.Blue, 2));
            return distance < threshold;
        }
    }
}