/* Copyright 2017 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

package com.example.android.mlmodelscopepredictor;

import android.app.Activity;
import android.content.res.AssetFileDescriptor;
import android.content.res.AssetManager;
import android.graphics.Bitmap;
import android.os.Build;
import android.os.SystemClock;
import android.util.Log;
import java.io.BufferedReader;
import java.io.File;
import java.io.FileInputStream;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.io.OutputStream;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.MappedByteBuffer;
import java.nio.channels.FileChannel;
import java.util.AbstractMap;
import java.util.ArrayList;
import java.util.Comparator;
import java.util.List;
import java.util.Map;
import java.util.PriorityQueue;
//import org.tensorflow.lite.Interpreter;
import tflite.Tflite;
import tflite.PredictorData;

/** Classifies images with Tensorflow Lite. */
public class ImageClassifier {

  /** Tag for the {@link Log}. */
  private static final String TAG = "MLModelScopePredictor";

  /** Name of the model file stored in Assets. */
  private static final String MODEL_PATH = "graph.lite";


  /** Name of the label file stored in Assets. */
  private static final String LABEL_PATH = "labels.txt";
  public String LABEL_PATH_LOCAL;

  /** Number of results to show in the UI. */
  private static final int RESULTS_TO_SHOW = 3;

  /** Dimensions of inputs. */
  private static final int DIM_BATCH_SIZE = 1;

  private static final int DIM_PIXEL_SIZE = 3;

  static final int DIM_IMG_SIZE_X = 224;
  static final int DIM_IMG_SIZE_Y = 224;

  private static final int IMAGE_MEAN = 128;
  private static final float IMAGE_STD = 128.0f;


  /* Preallocated buffers for storing image data in. */
  private int[] intValues = new int[DIM_IMG_SIZE_X * DIM_IMG_SIZE_Y];

  /* DEMO: An instance of the driver class to run model inference with Tensorflow Lite. */
  //private Interpreter tflite;
  /** Define our predictor constructor */
  private PredictorData mypredictor;

  /** Labels corresponding to the output of the vision model. */
  private List<String> labelList;

  /** A ByteBuffer to hold image data, to be feed into Tensorflow Lite as inputs. */
  private ByteBuffer imgData = null;

  /** An array to hold inference results, to be feed into Tensorflow Lite as outputs. */
  private float[][] labelProbArray = null;
  /** multi-stage low pass filter **/
  private float[][] filterLabelProbArray = null;
  private static final int FILTER_STAGES = 3;
  private static final float FILTER_FACTOR = 0.4f;

  private PriorityQueue<Map.Entry<String, Float>> sortedLabels =
      new PriorityQueue<>(
          RESULTS_TO_SHOW,
          new Comparator<Map.Entry<String, Float>>() {
            @Override
            public int compare(Map.Entry<String, Float> o1, Map.Entry<String, Float> o2) {
              return (o1.getValue()).compareTo(o2.getValue());
            }
          });

  /* DEMO: Initializes an {@code ImageClassifier}.
  ImageClassifier(Activity activity) throws IOException {
    tflite = new Interpreter(loadModelFile(activity));
    labelList = loadLabelList(activity);
    imgData =
        ByteBuffer.allocateDirect(
            4 * DIM_BATCH_SIZE * DIM_IMG_SIZE_X * DIM_IMG_SIZE_Y * DIM_PIXEL_SIZE);
    imgData.order(ByteOrder.nativeOrder());
    labelProbArray = new float[1][labelList.size()];
    filterLabelProbArray = new float[FILTER_STAGES][labelList.size()];
    Log.d(TAG, "Created a Tensorflow Lite Image Classifier.");
  }*/

  ImageClassifier(Activity activity) throws IOException {
    try{
      // DOES NOT WORK read tflite graph into a buffer
      //AssetFileDescriptor fd = activity.getAssets().openFd(MODEL_PATH);
      //FileInputStream is = new FileInputStream(fd.getFileDescriptor());
      //int is_size = is.available();
      //byte[] is_buffer = new byte[is_size];
      //is.read(is_buffer);
      //is.close();

      // TRY temporary storage
      AssetManager assetManager = activity.getAssets();
      String abi = Build.CPU_ABI;
      String filesDir = activity.getFilesDir().getPath();
      String testPath = abi + "/" + MODEL_PATH;
      String testPathLabels = abi + "/" + LABEL_PATH;

      InputStream inStream = assetManager.open(MODEL_PATH);
      Log.d(TAG, "Opened" + MODEL_PATH);
      InputStream inStreamLabels = assetManager.open(LABEL_PATH);
      Log.d(TAG, "Opened" + LABEL_PATH);

      // Copy this file to an executable location
      File outFile = new File(filesDir, MODEL_PATH);
      File outFileLabels = new File(filesDir, LABEL_PATH);

      OutputStream outStream = new FileOutputStream(outFile);
      OutputStream outStreamLabels = new FileOutputStream(outFileLabels);

      byte[] buffer = new byte[1024];
      int read;
      while ((read = inStream.read(buffer)) != -1){
        outStream.write(buffer, 0, read);
      }
      byte[] bufferLabels = new byte[1024];
      int readLabels;
      while ((readLabels = inStreamLabels.read(bufferLabels)) != -1){
        outStreamLabels.write(bufferLabels, 0, readLabels);
      }

      inStream.close();
      outStream.flush();
      outStream.close();
      Log.d(TAG, "Copied" + MODEL_PATH + " to " + filesDir);
      String tempPath = filesDir + "/" + MODEL_PATH;

      inStreamLabels.close();
      outStreamLabels.flush();
      outStreamLabels.close();
      Log.d(TAG, "Copied" + LABEL_PATH + " to " + filesDir);
      String tempPathLabels = filesDir + "/" + LABEL_PATH;
      LABEL_PATH_LOCAL = tempPathLabels;

      mypredictor = Tflite.new_(tempPath, Tflite.CPUMode, 1);
      if(mypredictor == null){
        Log.e(TAG, "Tflite.new_ returning null model");
      }
    }catch (Exception e){
      e.printStackTrace();
    }
    labelList = loadLabelList(activity);
    imgData =
            ByteBuffer.allocateDirect(
                    4 * DIM_BATCH_SIZE * DIM_IMG_SIZE_X * DIM_IMG_SIZE_Y * DIM_PIXEL_SIZE);
    imgData.order(ByteOrder.nativeOrder());
    labelProbArray = new float[1][labelList.size()];
    filterLabelProbArray = new float[FILTER_STAGES][labelList.size()];
    Log.d(TAG, "Created a Tensorflow Lite Image Classifier.");
  }

  /* DEMO: Classifies a frame from the preview stream.
  String classifyFrame(Bitmap bitmap) {
    if (tflite == null) {
      Log.e(TAG, "Image classifier has not been initialized; Skipped.");
      return "Uninitialized Classifier.";
    }
    convertBitmapToByteBuffer(bitmap);
    // Here's where the magic happens!!!
    long startTime = SystemClock.uptimeMillis();
    tflite.run(imgData, labelProbArray);
    long endTime = SystemClock.uptimeMillis();
    Log.d(TAG, "Timecost to run model inference: " + Long.toString(endTime - startTime));

    // smooth the results
    applyFilter();

    // print the results
    String textToShow = printTopKLabels();
    textToShow = Long.toString(endTime - startTime) + "ms" + textToShow;
    return textToShow;
  }*/

  String classifyFrame(Bitmap bitmap) {
    if (mypredictor == null) {
      Log.e(TAG, "Image classifier has not been initialized; Skipped.");
      return "Uninitialized Classifier.";
    }
    // read bitmapped frame into imgData (ByteBuffer)
    convertBitmapToByteBuffer(bitmap);
    // convert ByteBuffer[] into byte[]
    // as gomobile only supports []byte
    imgData.rewind();
    byte[] imgDataBytes = new byte[imgData.remaining()];
    try {

      // DEBUG - NOT PRINTING => meaning there is no corruption of data
      if(imgData.getFloat(2) == 0.0){
        Log.d(TAG,"imgData is null - WHY ?????");
      }

      imgData.get(imgDataBytes, 0, imgDataBytes.length);

      // DEBUG - NOT PRINTING => meaning imgDataBytes were transferred correctly
      if(imgDataBytes.length != 0){
        Log.d(TAG,"imgDataBytes length = " + Float.toString(imgDataBytes.length));
      }

    }catch (Exception e){
      e.printStackTrace();
    }

    // Here's where the magic happens!!!
    long startTime = SystemClock.uptimeMillis();
    // DEBUG - COMMENTING BOTH predict() and readPredictionOutput()
    // does not result in an error
    // try uncommenting one of them (first one)
    try {
      Tflite.predict(mypredictor, imgDataBytes);
    }catch(Exception e){
      e.printStackTrace();
    }
    long endTime = SystemClock.uptimeMillis();
    Log.d(TAG, "Timecost to run model inference: " + Long.toString(endTime - startTime));

    // smooth the results
    //applyFilter();

    // print the results
    //String textToShow = printTopKLabels();
    String labelOutput = "";
    try {
      Log.d(TAG, "CALLING readPredictedOutput");
      labelOutput = Tflite.readPredictionOutput(mypredictor, LABEL_PATH_LOCAL);
    }catch(Exception e){
      e.printStackTrace();
    }

    String textToShow = " labelOutput: " + labelOutput;
    textToShow = Long.toString(endTime - startTime) + "ms" + textToShow;
    return textToShow;
  }

  void applyFilter(){
    int num_labels =  labelList.size();

    // Low pass filter `labelProbArray` into the first stage of the filter.
    for(int j=0; j<num_labels; ++j){
      filterLabelProbArray[0][j] += FILTER_FACTOR*(labelProbArray[0][j] -
                                                   filterLabelProbArray[0][j]);
    }
    // Low pass filter each stage into the next.
    for (int i=1; i<FILTER_STAGES; ++i){
      for(int j=0; j<num_labels; ++j){
        filterLabelProbArray[i][j] += FILTER_FACTOR*(
                filterLabelProbArray[i-1][j] -
                filterLabelProbArray[i][j]);

      }
    }

    // Copy the last stage filter output back to `labelProbArray`.
    for(int j=0; j<num_labels; ++j){
      labelProbArray[0][j] = filterLabelProbArray[FILTER_STAGES-1][j];
    }
  }

  /* DEMO: Closes tflite to release resources.
  public void close() {
    tflite.close();
    tflite = null;
  }*/

  public void close() {
    Tflite.close(mypredictor);
  }

  /** Reads label list from Assets. */
  private List<String> loadLabelList(Activity activity) throws IOException {
    List<String> labelList = new ArrayList<String>();
    BufferedReader reader =
        new BufferedReader(new InputStreamReader(activity.getAssets().open(LABEL_PATH)));
    String line;
    while ((line = reader.readLine()) != null) {
      labelList.add(line);
    }
    reader.close();
    return labelList;
  }

  /** Memory-map the model file in Assets. */
  private MappedByteBuffer loadModelFile(Activity activity) throws IOException {
    AssetFileDescriptor fileDescriptor = activity.getAssets().openFd(MODEL_PATH);
    FileInputStream inputStream = new FileInputStream(fileDescriptor.getFileDescriptor());
    FileChannel fileChannel = inputStream.getChannel();
    long startOffset = fileDescriptor.getStartOffset();
    long declaredLength = fileDescriptor.getDeclaredLength();
    return fileChannel.map(FileChannel.MapMode.READ_ONLY, startOffset, declaredLength);
  }

  /** Writes Image data into a {@code ByteBuffer}. */
  private void convertBitmapToByteBuffer(Bitmap bitmap) {
    if (imgData == null) {
      return;
    }
    imgData.rewind();

    // check if passed bitmap is empty
    Bitmap emptyBitmap = Bitmap.createBitmap(bitmap.getWidth(), bitmap.getHeight(), bitmap.getConfig());
    if(bitmap.sameAs(emptyBitmap)){
      Log.d(TAG, "Passed bitmap is also empty - what is happening ???");
    }
    
    bitmap.getPixels(intValues, 0, bitmap.getWidth(), 0, 0, bitmap.getWidth(), bitmap.getHeight());
    // Convert the image to floating point.
    int pixel = 0;
    long startTime = SystemClock.uptimeMillis();
    for (int i = 0; i < DIM_IMG_SIZE_X; ++i) {
      for (int j = 0; j < DIM_IMG_SIZE_Y; ++j) {
        final int val = intValues[pixel++];

        // DEBUG - NOT PRINTING => intValues is fine
        if(val == 0)
          Log.d(TAG, "val[" + Long.toString(pixel-1) + "] is zero");

        imgData.putFloat((((val >> 16) & 0xFF)-IMAGE_MEAN)/IMAGE_STD);
        imgData.putFloat((((val >> 8) & 0xFF)-IMAGE_MEAN)/IMAGE_STD);
        imgData.putFloat((((val) & 0xFF)-IMAGE_MEAN)/IMAGE_STD);
      }
    }
    long endTime = SystemClock.uptimeMillis();
    Log.d(TAG, "Timecost to put values into ByteBuffer: " + Long.toString(endTime - startTime));
    if (imgData.getFloat(2) != 0.0){
      Log.d(TAG, "Non-zero imgData value - " + Float.toString(imgData.getFloat(2)));
    }
  }

  /** Prints top-K labels, to be shown in UI as the results. */
  private String printTopKLabels() {
    for (int i = 0; i < labelList.size(); ++i) {
      sortedLabels.add(
          new AbstractMap.SimpleEntry<>(labelList.get(i), labelProbArray[0][i]));
      if (sortedLabels.size() > RESULTS_TO_SHOW) {
        sortedLabels.poll();
      }
    }
    String textToShow = "";
    final int size = sortedLabels.size();
    for (int i = 0; i < size; ++i) {
      Map.Entry<String, Float> label = sortedLabels.poll();
      textToShow = String.format("\n%s: %4.2f",label.getKey(),label.getValue()) + textToShow;
    }
    return textToShow;
  }
}
