package com.example.handrecognition;

import android.content.res.AssetFileDescriptor;
import android.content.res.AssetManager;
import android.graphics.Bitmap;
import android.util.Log;
import android.view.View;
import android.widget.Button;
import android.widget.TextView;
import android.widget.Toast;

import androidx.annotation.NonNull;

import org.opencv.android.Utils;
import org.opencv.core.Core;
import org.opencv.core.Mat;
import org.opencv.core.Point;
import org.opencv.core.Rect;
import org.opencv.core.Scalar;
import org.opencv.imgproc.Imgproc;
import org.tensorflow.lite.Interpreter;
import org.tensorflow.lite.gpu.GpuDelegate;

import java.io.FileInputStream;
import java.io.IOException;
import java.lang.reflect.Array;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.channels.FileChannel;
import java.util.Map;
import java.util.TreeMap;

public class signLanguageClass {
    // Interpreter for the hand detection model
    private final Interpreter interpreter;
    // Interpreter for the ASL recognition model
    private final Interpreter aslInterpreter;
    // Input size for the hand detection model
    private final int INPUT_SIZE;
    // Input size for the ASL recognition model
    private int aslInputSize = 0;
    // String to store the combined letters
    private String combine_letters = "";
    // String to store the current letter
    private String current_letter = "";

    // Constructor for the signLanguageClass
    signLanguageClass(Button add_letter_button, Button backspace_button, Button space_button,
                      TextView combine_letters_text_view, AssetManager assetManager, String modelPath,
                      int inputSize, String aslModelPath, int aslInputSize) throws IOException {
        // Set the input sizes for the models
        INPUT_SIZE = inputSize;
        this.aslInputSize = aslInputSize;

        // Initialize the interpreters with the models
        // Options for the hand detection model
        Interpreter.Options options = new Interpreter.Options();
        // Use GPU delegate for the hand detection model
        GpuDelegate gpuDelegate = new GpuDelegate();
        options.addDelegate(gpuDelegate);
        // Load the hand detection model
        interpreter = new Interpreter(loadModelFile(assetManager, modelPath), options);

        // Options for the ASL recognition model
        Interpreter.Options aslOptions = new Interpreter.Options();
        // Use all available threads for the ASL recognition model
        aslOptions.setNumThreads(-1);
        // Load the ASL recognition model
        aslInterpreter = new Interpreter(loadModelFile(assetManager, aslModelPath), aslOptions);

        // Set up the buttons' onClickListeners
        // Add the current letter to the combined letters when the add letter button is clicked
        add_letter_button.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View view) {
                combine_letters += current_letter;
                combine_letters_text_view.setText(combine_letters);
            }
        });

        // Remove the last letter from the combined letters when the backspace button is clicked
        backspace_button.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View view) {
                if (combine_letters.isEmpty()) {
                    combine_letters_text_view.setText("");
                    Toast.makeText(view.getContext(), "No letters to delete", Toast.LENGTH_SHORT).show();
                } else {
                    combine_letters = combine_letters.substring(0, combine_letters.length() - 1);
                    combine_letters_text_view.setText(combine_letters);
                }
            }
        });

        // Add a space to the combined letters when the space button is clicked
        space_button.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View view) {
                combine_letters += " ";
                combine_letters_text_view.setText(combine_letters);
            }
        });
    }

    // Method to create a map for the output of the model
    @NonNull
    private static Map<Integer, Object> getIntegerObjectMap() {
        // Create a TreeMap to store the output of the model
        Map<Integer, Object> output_map = new TreeMap<>();
        // Array to store the bounding boxes of the detected objects
        float[][][] boxes = new float[1][10][4];
        // Array to store the scores of the detected objects
        float[][] scores = new float[1][10];
        // Array to store the classes of the detected objects
        float[][] classes = new float[1][10];
        // Add the arrays to the output map
        output_map.put(0, boxes);
        output_map.put(1, classes);
        output_map.put(2, scores);
        return output_map;
    }

    // Method to load the model file
    private ByteBuffer loadModelFile(AssetManager assetManager, String modelPath) throws IOException {
        // Open the model file
        AssetFileDescriptor fileDescriptor = assetManager.openFd(modelPath);
        FileInputStream inputStream = new FileInputStream(fileDescriptor.getFileDescriptor());
        FileChannel fileChannel = inputStream.getChannel();
        long startOffset = fileDescriptor.getStartOffset();
        long declaredLength = fileDescriptor.getDeclaredLength();

        // Map the model file into a ByteBuffer
        return fileChannel.map(FileChannel.MapMode.READ_ONLY, startOffset, declaredLength);
    }

    // Method to recognize the image
    public Mat recognizeImage(Mat mat_image) {
        // Rotate the image to portrait orientation
        Mat rotated_mat_image = new Mat();
        Mat a = mat_image.t();
        Core.flip(a, rotated_mat_image, 1);
        a.release();

        // Convert the Mat to a Bitmap
        Bitmap bitmap = Bitmap.createBitmap(rotated_mat_image.cols(), rotated_mat_image.rows(), Bitmap.Config.ARGB_8888);
        Utils.matToBitmap(rotated_mat_image, bitmap);
        int height = bitmap.getHeight();
        int width = bitmap.getWidth();

        // Scale the Bitmap to the input size of the model
        Bitmap scaledBitmap = Bitmap.createScaledBitmap(bitmap, INPUT_SIZE, INPUT_SIZE, false);

        // Convert the Bitmap to a ByteBuffer
        ByteBuffer byteBuffer = convertBitmapToByteBuffer(scaledBitmap);

        // Create the input for the model
        Object[] input = new Object[1];
        input[0] = byteBuffer;

        // Create the output map for the model
        Map<Integer, Object> output_map = getIntegerObjectMap();

        // Run the model
        interpreter.runForMultipleInputsOutputs(input, output_map);

        // Get the output of the model
        Object value = output_map.get(0);
        Object Object_class = output_map.get(1);
        Object score = output_map.get(2);

        // Loop through each detected object
        for (int i = 0; i < 10; i++) {
            float class_value = (float) Array.get(Array.get(Object_class, 0), i);
            float score_value = (float) Array.get(Array.get(score, 0), i);
            // If the score is above the threshold
            if (score_value > 0.9f) {
                // Get the bounding box of the object
                Object box1 = Array.get(Array.get(value, 0), i);
                float y1 = (float) Array.get(box1, 0) * height;
                float x1 = (float) Array.get(box1, 1) * width;
                float y2 = (float) Array.get(box1, 2) * height;
                float x2 = (float) Array.get(box1, 3) * width;

                // Ensure the bounding box is within the image
                if (y1 < 0) {
                    y1 = 0;
                }
                if (x1 < 0) {
                    x1 = 0;
                }
                if (y2 > height) {
                    y2 = height;
                }
                if (x2 > width) {
                    x2 = width;
                }

                // Get the width and height of the bounding box
                float width_roi = x2 - x1;
                float height_roi = y2 - y1;

                // Crop the hand image from the original frame
                Rect cropped_roi = new Rect((int) x1, (int) y1, (int) width_roi, (int) height_roi);
                Mat cropped_image = new Mat(rotated_mat_image, cropped_roi).clone();

                // Convert the cropped image to a Bitmap
                Bitmap cropped_bitmap = Bitmap.createBitmap(cropped_image.cols(), cropped_image.rows(), Bitmap.Config.ARGB_8888);
                Utils.matToBitmap(cropped_image, cropped_bitmap);

                // Resize the cropped Bitmap to the input size of the ASL model
                Bitmap resized_bitmap = Bitmap.createScaledBitmap(cropped_bitmap, aslInputSize, aslInputSize, false);

                // Convert the resized Bitmap to a ByteBuffer
                ByteBuffer aslByteBuffer = convertBitmapToByteBufferAsl(resized_bitmap);

                // Create the output for the ASL model
                float[][] aslOutput = new float[1][1];

                // Run the ASL model
                aslInterpreter.run(aslByteBuffer, aslOutput);

                // Convert the output of the ASL model to a sign
                String aslSign = getSign(aslOutput[0][0]);

                // Log the sign recognized by the ASL model to the console for debugging purposes
                Log.d("ASL Sign", "ASL Sign: " + aslSign);

                // Set the current letter
                current_letter = aslSign;

                // Add the sign to the image
                Imgproc.putText(rotated_mat_image, aslSign, new Point(x1 + 10, y1 + 40), 3, 2, new Scalar(0, 255, 0, 255), 2);

                // Draw the bounding box on the image
                Imgproc.rectangle(rotated_mat_image, new Point(x1, y1), new Point(x2, y2), new Scalar(0, 255, 0, 255), 2);
            }
        }

        // Rotate the image back to the original orientation
        Mat b = rotated_mat_image.t();
        Core.flip(b, mat_image, 0);
        b.release();

        // Return the image with the bounding boxes and signs
        return mat_image;
    }

    // Method to convert the output of the ASL model to a sign
    private String getSign(float value) {
        String valueString;
        // Convert the output value to a sign
        if (value >= -0.5 & value < 0.5) {
            valueString = "A";
        } else if (value >= 0.5 & value < 1.5) {
            valueString = "B";
        } else if (value >= 1.5 & value < 2.5) {
            valueString = "C";
        } else if (value >= 2.5 & value < 3.5) {
            valueString = "D";
        } else if (value >= 3.5 & value < 4.5) {
            valueString = "E";
        } else if (value >= 4.5 & value < 5.5) {
            valueString = "F";
        } else if (value >= 5.5 & value < 6.5) {
            valueString = "G";
        } else if (value >= 6.5 & value < 7.5) {
            valueString = "H";
        } else if (value >= 7.5 & value < 8.5) {
            valueString = "I";
        } else if (value >= 8.5 & value < 9.5) {
            valueString = "J";
        } else if (value >= 9.5 & value < 10.5) {
            valueString = "K";
        } else if (value >= 10.5 & value < 11.5) {
            valueString = "L";
        } else if (value >= 11.5 & value < 12.5) {
            valueString = "M";
        } else if (value >= 12.5 & value < 13.5) {
            valueString = "N";
        } else if (value >= 13.5 & value < 14.5) {
            valueString = "O";
        } else if (value >= 14.5 & value < 15.5) {
            valueString = "P";
        } else if (value >= 15.5 & value < 16.5) {
            valueString = "Q";
        } else if (value >= 16.5 & value < 17.5) {
            valueString = "R";
        } else if (value >= 17.5 & value < 18.5) {
            valueString = "S";
        } else if (value >= 18.5 & value < 19.5) {
            valueString = "T";
        } else if (value >= 19.5 & value < 20.5) {
            valueString = "U";
        } else if (value >= 20.5 & value < 21.5) {
            valueString = "V";
        } else if (value >= 21.5 & value < 22.5) {
            valueString = "W";
        } else if (value >= 22.5 & value < 23.5) {
            valueString = "X";
        } else if (value >= 23.5 & value < 24.5) {
            valueString = "Y";
        } else {
            valueString = "Invalid";
        }

        return valueString;
    }

    // Method to convert a bitmap to a ByteBuffer
    private ByteBuffer convertBitmapToByteBuffer(Bitmap bitmap) {
        ByteBuffer byteBuffer;
        int size_images = INPUT_SIZE;
        // Allocate a direct ByteBuffer of size 4 * size_images * size_images * 3
        byteBuffer = ByteBuffer.allocateDirect(4 * size_images * size_images * 3);
        byteBuffer.order(ByteOrder.nativeOrder());
        int[] intValues = new int[size_images * size_images];
        // Get the pixels from the bitmap
        bitmap.getPixels(intValues, 0, bitmap.getWidth(), 0, 0, bitmap.getWidth(), bitmap.getHeight());
        int pixel = 0;

        // Loop through each pixel and add it to the ByteBuffer
        for (int i = 0; i < size_images; ++i) {
            for (int j = 0; j < size_images; ++j) {
                final int val = intValues[pixel++];
                float IMAGE_STD = 255.0f;
                byteBuffer.putFloat((((val >> 16) & 0xFF)) / IMAGE_STD);
                byteBuffer.putFloat((((val >> 8) & 0xFF)) / IMAGE_STD);
                byteBuffer.putFloat((((val) & 0xFF)) / IMAGE_STD);
            }
        }
        return byteBuffer;
    }

    // Method to convert a bitmap to a ByteBuffer for the ASL model
    private ByteBuffer convertBitmapToByteBufferAsl(Bitmap bitmap) {
        ByteBuffer byteBuffer;
        int size_images = aslInputSize;
        // Allocate a direct ByteBuffer of size 4 * size_images * size_images * 3
        byteBuffer = ByteBuffer.allocateDirect(4 * size_images * size_images * 3);
        byteBuffer.order(ByteOrder.nativeOrder());
        int[] intValues = new int[size_images * size_images];
        // Get the pixels from the bitmap
        bitmap.getPixels(intValues, 0, bitmap.getWidth(), 0, 0, bitmap.getWidth(), bitmap.getHeight());
        int pixel = 0;

        // Loop through each pixel and add it to the ByteBuffer
        for (int i = 0; i < size_images; ++i) {
            for (int j = 0; j < size_images; ++j) {
                final int val = intValues[pixel++];
                byteBuffer.putFloat((((val >> 16) & 0xFF)));
                byteBuffer.putFloat((((val >> 8) & 0xFF)));
                byteBuffer.putFloat((((val) & 0xFF)));
            }
        }
        return byteBuffer;
    }
}
