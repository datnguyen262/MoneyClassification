#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <iostream>

using namespace cv;
using namespace std;


Point Matching(int, void*, Mat img, Mat keypoint)
{

    Mat result;
    Mat img_display;
    img.copyTo(img_display);
    int result_cols = img.cols - keypoint.cols + 1;
    int result_rows = img.rows - keypoint.rows + 1;
    result.create(result_rows, result_cols, CV_32FC1);
    matchTemplate(img, keypoint, result, TM_CCOEFF_NORMED);
    normalize(result, result, 0, 1, NORM_MINMAX, -1, Mat());
    double minVal; double maxVal; Point minLoc; Point maxLoc;
    Point match;
    minMaxLoc(result, &minVal, &maxVal, &minLoc, &maxLoc, Mat());
    match = maxLoc;
    rectangle(img_display, match, Point(match.x + keypoint.cols, match.y + keypoint.rows), Scalar::all(0), 2, 8, 0);
    return match;
}

int main()
{

    //Đọc ảnh đầu vào
    Mat input = imread("tienban.png", IMREAD_COLOR);
    // Chuyển ảnh sang màu xám để sử lý
    Mat input_gray;
    cvtColor(input, input_gray, COLOR_BGR2GRAY);
    // Sử dụng Gaussian BLur để làm mịn ảnh
    Mat blur;
    GaussianBlur(input_gray, blur, Size(3, 3), 0);
    // Dùng hàm threhold để tách đối tượng khỏi nền
    Mat thresholded;
    threshold(blur, thresholded, 0, 255, THRESH_BINARY);
    //sử dụng hàm findContours để tìm các đường viền trong ảnh
    vector<vector<Point>> contours;
    vector<Vec4i> hierarchy;
    Mat contourImage = Mat::zeros(thresholded.size(), CV_8UC3);
    findContours(thresholded, contours, hierarchy, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);
    //Duyệt và tìm các đường viền có diện tích lớn nhất, xác định nó và đó là khung ảnh
    double max_area = 0;
    int max_area_contour = -1;
    for (int i = 0; i < contours.size(); i++) {
        double area = contourArea(contours[i]);
        if (area > max_area) {
            max_area = area;
            max_area_contour = i;
        }
    }

    if (max_area_contour >= 0) {

        Rect bounding_rect = boundingRect(contours[max_area_contour]);
        Mat cropped_image = input(bounding_rect);
        rectangle(input, bounding_rect, Scalar(0, 0, 255), 2);

        //Resize hình ảnh đã cắt xong
        Mat resized_image;
        resize(cropped_image, resized_image, Size(800, 349)); // 800x349 là kích thước mới của ảnh
        imwrite("cropped_image.png", resized_image);

        // Đọc ảnh mẫu vs ảnh bẩn đã cắt
        Mat image1 = imread("tienmau.png");
        Mat image2 = imread("cropped_image.png");

        // Chuyển đổi ảnh sang định dạng YCrCb
        Mat ycrcb1, ycrcb2;
        cvtColor(image1, ycrcb1, COLOR_BGR2YCrCb);
        cvtColor(image2, ycrcb2, COLOR_BGR2YCrCb);

        // Tách ra từng kênh màu Y, Cr, Cb
        vector<Mat> color1, color2;
        split(ycrcb1, color1);
        split(ycrcb2, color2);

        // Cân bằng độ sáng của kênh màu
        equalizeHist(color1[0], color1[0]);
        equalizeHist(color2[0], color2[0]);
        /*     equalizeHist(color1[1], color1[1]);
             equalizeHist(color2[1], color2[1]);
             equalizeHist(color1[2], color1[2]);
             equalizeHist(color2[2], color2[2]);*/

             // Ghép lại các kênh màu Y, Cr, Cb
        Mat fin_ycrcb1, fin_ycrcb2;
        merge(color1, fin_ycrcb1);
        merge(color2, fin_ycrcb2);

        // Chuyển đổi lại sang định dạng RGB
        Mat fin_image1, fin_image2;
        cvtColor(fin_ycrcb1, fin_image1, COLOR_YCrCb2BGR);
        cvtColor(fin_ycrcb2, fin_image2, COLOR_YCrCb2BGR);

        // Hiển thị ảnh đã cân bằng độ sáng
        imwrite("equalized_image_1.png", fin_image1);
        imwrite("equalized_image_2.png", fin_image2);

        // Tìm bẩn
        //tải ảnh chứa điểm trọng tâm 
        Mat point = imread("tem.png");
        //lưu trữ tọa độ của điểm gốc và điểm chụp trên ảnh
        Point origin = Matching(0, 0, fin_image1, point);
        Point capture = Matching(0, 0, fin_image2, point);

        //tính toán sự thay đổi vị trí giữa 2 ảnh
        int delta_x = capture.x - origin.x;
        int delta_y = capture.y - origin.y;
        cout << delta_x;
        Mat virtual_image = fin_image1.clone();
        for (int y = 0; y < fin_image1.rows; y++)
        {
            for (int x = 0; x < fin_image1.cols * 3; x++)
            {

                virtual_image.at<uchar>(y, x) = fin_image2.at<uchar>(y + delta_y, x + delta_x * 3);
            }
        }

        Mat minus = abs(fin_image1 - virtual_image);
        imwrite("Minus_image.png", minus);


        //Bộ lọc Segmentation

        Mat minus_gray(minus.rows, minus.cols, CV_8UC1);
        cvtColor(minus, minus_gray, COLOR_BGR2GRAY);

        //Áp dụng phân ngưỡng để tách vật thể và nền
        Mat binary(minus.rows, minus.cols, CV_8UC1);
        threshold(minus_gray, binary, 40, 255, THRESH_BINARY | THRESH_OTSU);

        //Tạo mặt nạ cho nền bằng phép co dãn (erode)
        Mat mark_erode;
        erode(binary, mark_erode, Mat(), Point(-1, -1), 2);

        //Tạo mặt nạ cho nền bằng phép giãn (dilate)
        Mat mark_dilate;
        dilate(binary, mark_dilate, Mat(), Point(-1, -1), 3);
        threshold(mark_dilate, mark_dilate, 1, 128, THRESH_BINARY_INV);

        //Tạo đánh dấu(marker) bằng cách kết hợp mặt nạ cho vật thể và nền
        Mat marker(binary.size(), CV_8U, Scalar(0));
        marker = mark_erode + mark_dilate;
        marker.convertTo(marker, CV_32S);

        //Áp dụng phương pháp watershed
        watershed(minus, marker);
        marker.convertTo(marker, CV_8U);

        //Áp dụng phân ngưỡng để phân loại ảnh thành vật thể và nền
        threshold(marker, marker, 40, 255, THRESH_BINARY | THRESH_OTSU);

        imshow("Result", marker);
        imwrite("Result.png", marker);


    }
    waitKey(0);
    return 0;
}