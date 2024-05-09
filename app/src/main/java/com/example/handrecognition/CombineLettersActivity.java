package com.example.handrecognition;

import androidx.core.app.ActivityCompat;
import androidx.core.content.ContextCompat;

import android.Manifest;
import android.app.Activity;
import android.content.pm.PackageManager;
import android.os.Bundle;
import android.text.method.ScrollingMovementMethod;
import android.util.Log;
import android.view.SurfaceView;
import android.view.Window;
import android.view.WindowManager;
import android.widget.Button;
import android.widget.TextView;

import org.opencv.android.BaseLoaderCallback;
import org.opencv.android.CameraBridgeViewBase;
import org.opencv.android.LoaderCallbackInterface;
import org.opencv.android.OpenCVLoader;
import org.opencv.core.CvType;
import org.opencv.core.Mat;

import java.io.IOException;

public class CombineLettersActivity extends Activity implements CameraBridgeViewBase.CvCameraViewListener2 {
    private static final String TAG = "MainActivity";

    // Matrices for storing image data
    private Mat mRgba;

    // OpenCV camera view
    private CameraBridgeViewBase mOpenCvCameraView;

    // Callback for OpenCV manager connections
    private final BaseLoaderCallback mLoaderCallback = new BaseLoaderCallback(this) {
        @Override
        public void onManagerConnected(int status) {
            if (status == LoaderCallbackInterface
                    .SUCCESS) {
                Log.i(TAG, "OpenCv Is loaded");
                mOpenCvCameraView.enableView();
            }
            super.onManagerConnected(status);
        }
    };

    // Instance of the sign language class
    private signLanguageClass signLanguageClass;

    public CombineLettersActivity() {
        Log.i(TAG, "Instantiated new " + this.getClass());
    }

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        requestWindowFeature(Window.FEATURE_NO_TITLE);
        getWindow().addFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON);

        // Request camera permissions
        int MY_PERMISSIONS_REQUEST_CAMERA = 0;
        if (ContextCompat.checkSelfPermission(CombineLettersActivity.this, Manifest.permission.CAMERA)
                == PackageManager.PERMISSION_DENIED) {
            ActivityCompat.requestPermissions(CombineLettersActivity.this,
                    new String[]{Manifest.permission.CAMERA}, MY_PERMISSIONS_REQUEST_CAMERA);
        }

        setContentView(R.layout.activity_combine_letters);

        // Initialize OpenCV camera view
        mOpenCvCameraView = findViewById(R.id.frame_Surface);
        mOpenCvCameraView.setVisibility(SurfaceView.VISIBLE);
        mOpenCvCameraView.setCvCameraViewListener(this);

        // Initialize UI elements
        Button add_letter_button = findViewById(R.id.add_text_button);
        Button backspace_button = findViewById(R.id.backspace_button);
        Button space_button = findViewById(R.id.space_button);
        TextView combine_letters_text_view = findViewById(R.id.combine_letters_text_view);
        combine_letters_text_view.setMovementMethod(new ScrollingMovementMethod());

        // Initialize sign language class
        try {
            signLanguageClass = new signLanguageClass(add_letter_button, backspace_button,
                    space_button, combine_letters_text_view, getAssets(), "hand_modelv2.tflite",
                    300, "model.tflite", 96);
            Log.d("MainActivity", "Model is successfully loaded");
        } catch (IOException e) {
            Log.d("MainActivity", "Getting some error");
            e.printStackTrace();
        }
    }

    @Override
    protected void onResume() {
        super.onResume();

        // Check if the camera permission has been granted
        if (ContextCompat.checkSelfPermission(CombineLettersActivity.this, Manifest.permission.CAMERA)
                == PackageManager.PERMISSION_GRANTED) {
            // Initialize OpenCV
            if (OpenCVLoader.initDebug()) {
                Log.d(TAG, "Opencv initialization is done");
                mLoaderCallback.onManagerConnected(LoaderCallbackInterface.SUCCESS);
            } else {
                Log.d(TAG, "Opencv is not loaded. try again");
                OpenCVLoader.initAsync(OpenCVLoader.OPENCV_VERSION, this, mLoaderCallback);
            }
        } else {
            // Request camera permissions
            int MY_PERMISSIONS_REQUEST_CAMERA = 0;
            ActivityCompat.requestPermissions(CombineLettersActivity.this, new String[]{Manifest.permission.CAMERA}, MY_PERMISSIONS_REQUEST_CAMERA);
        }
    }

    @Override
    protected void onPause() {
        super.onPause();

        // Disable camera view on pause
        if (mOpenCvCameraView != null) {
            mOpenCvCameraView.disableView();
        }
    }

    public void onDestroy() {
        super.onDestroy();

        // Disable camera view on destroy
        if (mOpenCvCameraView != null) {
            mOpenCvCameraView.disableView();
        }

    }

    public void onCameraViewStarted(int width, int height) {
        // Initialize matrices with the size of the camera view
        mRgba = new Mat(height, width, CvType.CV_8UC4);
    }

    public void onCameraViewStopped() {
        // Release the RGBA matrix
        mRgba.release();
    }

    public Mat onCameraFrame(CameraBridgeViewBase.CvCameraViewFrame inputFrame) {
        // Get the RGBA and grayscale frames
        mRgba = inputFrame.rgba();

        // Recognize the image
        return signLanguageClass.recognizeImage(mRgba);
    }
}