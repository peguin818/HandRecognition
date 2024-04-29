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
    private final Interpreter interpreter;
    private final Interpreter aslInterpreter;
    private final int INPUT_SIZE;
    private int aslInputSize = 0;
    private String combine_letters = "";
    private String current_letter = "";

    signLanguageClass(Button add_letter_button, Button backspace_button, Button space_button,
                      TextView combine_letters_text_view, AssetManager assetManager, String modelPath,
                      int inputSize, String aslModelPath, int aslInputSize) throws IOException {
        INPUT_SIZE = inputSize;
        this.aslInputSize = aslInputSize;
        // use to define gpu or cpu and no. of threads
        Interpreter.Options options = new Interpreter.Options();
        // Used to initialize gpu in app
        GpuDelegate gpuDelegate = new GpuDelegate();
        options.addDelegate(gpuDelegate);
        // load model
        interpreter = new Interpreter(loadModelFile(assetManager, modelPath), options);
        // use to define gpu or cpu and no. of threads
        Interpreter.Options aslOptions = new Interpreter.Options();
        aslOptions.setNumThreads(-1);
        // load model
        aslInterpreter = new Interpreter(loadModelFile(assetManager, aslModelPath), aslOptions);

        add_letter_button.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View view) {
                combine_letters += current_letter;
                combine_letters_text_view.setText(combine_letters);
            }
        });

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

        space_button.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View view) {
                combine_letters += " ";
                combine_letters_text_view.setText(combine_letters);
            }
        });
    }

    @NonNull
    private static Map<Integer, Object> getIntegerObjectMap() {
        Map<Integer, Object> output_map = new TreeMap<>();
        // instead we create treemap of three array (boxes,score,classes)

        float[][][] boxes = new float[1][10][4];
        // 10: top 10 object detected
        // 4: coordinate in image
        float[][] scores = new float[1][10];
        // stores scores of 10 object
        float[][] classes = new float[1][10];
        // stores class of object

        // add it to object_map;
        output_map.put(0, boxes);
        output_map.put(1, classes);
        output_map.put(2, scores);
        return output_map;
    }

    private ByteBuffer loadModelFile(AssetManager assetManager, String modelPath) throws IOException {
        // use to get description of file
        AssetFileDescriptor fileDescriptor = assetManager.openFd(modelPath);
        FileInputStream inputStream = new FileInputStream(fileDescriptor.getFileDescriptor());
        FileChannel fileChannel = inputStream.getChannel();
        long startOffset = fileDescriptor.getStartOffset();
        long declaredLength = fileDescriptor.getDeclaredLength();

        return fileChannel.map(FileChannel.MapMode.READ_ONLY, startOffset, declaredLength);
    }

    // create new Mat function
    public Mat recognizeImage(Mat mat_image) {
        // Rotate original image by 90 degree get get portrait frame

        Mat rotated_mat_image = new Mat();

        Mat a = mat_image.t();
        Core.flip(a, rotated_mat_image, 1);
        // Release mat
        a.release();

        Bitmap bitmap = null;
        bitmap = Bitmap.createBitmap(rotated_mat_image.cols(), rotated_mat_image.rows(), Bitmap.Config.ARGB_8888);
        Utils.matToBitmap(rotated_mat_image, bitmap);
        // define height and width
        int height = bitmap.getHeight();
        int width = bitmap.getWidth();

        // scale the bitmap to input size of model
        Bitmap scaledBitmap = Bitmap.createScaledBitmap(bitmap, INPUT_SIZE, INPUT_SIZE, false);

        // convert bitmap to bytebuffer as model input should be in it
        ByteBuffer byteBuffer = convertBitmapToByteBuffer(scaledBitmap);

        Object[] input = new Object[1];
        input[0] = byteBuffer;

        Map<Integer, Object> output_map = getIntegerObjectMap();

        // now predict
        interpreter.runForMultipleInputsOutputs(input, output_map);

        Object value = output_map.get(0);
        Object Object_class = output_map.get(1);
        Object score = output_map.get(2);

        // loop through each object
        // as output has only 10 boxes
        for (int i = 0; i < 10; i++) {
            float class_value = (float) Array.get(Array.get(Object_class, 0), i);
            float score_value = (float) Array.get(Array.get(score, 0), i);
            // define threshold for score
            if (score_value > 0.9f) {
                Object box1 = Array.get(Array.get(value, 0), i);
                // we are multiplying it with Original height and width of frame
                float y1 = (float) Array.get(box1, 0) * height;
                float x1 = (float) Array.get(box1, 1) * width;
                float y2 = (float) Array.get(box1, 2) * height;
                float x2 = (float) Array.get(box1, 3) * width;

                // set boundary limit
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

                float width_roi = x2 - x1;
                float height_roi = y2 - y1;

                // crop hand image from original frame
                Rect cropped_roi = new Rect((int) x1, (int) y1, (int) width_roi, (int) height_roi);
                Mat cropped_image = new Mat(rotated_mat_image, cropped_roi).clone();

                // convert cropped image to bitmap
                Bitmap cropped_bitmap = Bitmap.createBitmap(cropped_image.cols(), cropped_image.rows()
                        , Bitmap.Config.ARGB_8888);
                Utils.matToBitmap(cropped_image, cropped_bitmap);

                // resize cropped bitmap to 96x96
                Bitmap resized_bitmap = Bitmap.createScaledBitmap(cropped_bitmap,
                        aslInputSize, aslInputSize, false);

                // convert resized bitmap to bytebuffer
                ByteBuffer aslByteBuffer = convertBitmapToByteBufferAsl(resized_bitmap);

                // create output array
                float[][] aslOutput = new float[1][1];

                // predict output
                aslInterpreter.run(aslByteBuffer, aslOutput);
                Log.d("ASL Output", String.valueOf(aslOutput[0][0]));

                // convert float to alphabets
                String aslSign = getSign(aslOutput[0][0]);

                // set current letter
                current_letter = aslSign;

                // add label to cropped image
                Imgproc.putText(rotated_mat_image, aslSign,
                        new Point(x1 + 10, y1 + 40), 3, 2, new Scalar(0, 255, 0, 255), 2);

                // draw rectangle in Original frame //  starting point    // ending point of box  // color of box       thickness
                Imgproc.rectangle(rotated_mat_image, new Point(x1, y1), new Point(x2, y2), new Scalar(0, 255, 0, 255), 2);
            }
        }
        // before returning rotate back by -90 degree
        Mat b = rotated_mat_image.t();
        Core.flip(b, mat_image, 0);
        b.release();

        return mat_image;
    }

    private String getSign(float value) {
        String valueString;
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

    private ByteBuffer convertBitmapToByteBuffer(Bitmap bitmap) {
        ByteBuffer byteBuffer;
        int size_images = INPUT_SIZE;
        byteBuffer = ByteBuffer.allocateDirect(4 * size_images * size_images * 3);
        byteBuffer.order(ByteOrder.nativeOrder());
        int[] intValues = new int[size_images * size_images];
        bitmap.getPixels(intValues, 0, bitmap.getWidth(), 0, 0, bitmap.getWidth(), bitmap.getHeight());
        int pixel = 0;

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

    private ByteBuffer convertBitmapToByteBufferAsl(Bitmap bitmap) {
        ByteBuffer byteBuffer;
        int size_images = aslInputSize;
        byteBuffer = ByteBuffer.allocateDirect(4 * size_images * size_images * 3);
        byteBuffer.order(ByteOrder.nativeOrder());
        int[] intValues = new int[size_images * size_images];
        bitmap.getPixels(intValues, 0, bitmap.getWidth(), 0, 0, bitmap.getWidth(), bitmap.getHeight());
        int pixel = 0;

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
