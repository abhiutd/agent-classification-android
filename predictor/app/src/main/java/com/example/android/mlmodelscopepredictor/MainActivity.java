package com.example.android.mlmodelscopepredictor;

import android.app.Activity;
import android.content.Intent;
import android.support.v7.app.AppCompatActivity;
import android.os.Bundle;
import android.view.View;
import android.widget.AdapterView;
import android.widget.ArrayAdapter;
import android.widget.Spinner;
import android.widget.Toast;

public class MainActivity extends Activity implements AdapterView.OnItemSelectedListener {

    private Spinner framework_spinner;
    private Spinner model_spinner;
    private Spinner hardware_spinner;
    private Spinner datatype_spinner;

    private static final String[] framework_paths = {"Tensorflow Lite"};
    private static final String[] model_paths = {"mobilenet_v1_0.5_160_quant.tflite", "mobilenet_v1_0.25_128.tflite", "mobilenet_v1_0.25_128_quant.tflite", "mobilenet_v1_1.0_224.tflite", "mobilenet_v2_1.0_224.tflite", "squeezenet.tflite", "densenet.tflite"};
    private static final String[] hardware_paths = {"CPU_1_thread", "CPU_2_thread", "CPU_3_thread", "CPU_4_thread", "CPU_5_thread", "CPU_6_thread", "GPU", "NNAPI"};
    private static final String[] datatype_paths = {"float32", "int8"};

    public String selected_framework = "XXX";
    public String selected_model = "XXX";
    public String selected_hardware = "XXX";
    public String selected_datatype = "XXX";

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);

        framework_spinner = (Spinner)findViewById(R.id.framework_spinner);
        ArrayAdapter<String> framework_adapter = new ArrayAdapter<String>(MainActivity.this, android.R.layout.simple_spinner_item, framework_paths);
        framework_adapter.setDropDownViewResource(android.R.layout.simple_spinner_dropdown_item);
        framework_spinner.setPrompt("Choose ML framework!");
        framework_spinner.setAdapter(framework_adapter);

        model_spinner = (Spinner)findViewById(R.id.model_spinner);
        ArrayAdapter<String> model_adapter = new ArrayAdapter<String>(MainActivity.this, android.R.layout.simple_spinner_item, model_paths);
        model_adapter.setDropDownViewResource(android.R.layout.simple_spinner_dropdown_item);
        model_spinner.setPrompt("Choose ML model!");
        model_spinner.setAdapter(model_adapter);

        hardware_spinner = (Spinner)findViewById(R.id.hardware_spinner);
        ArrayAdapter<String> hardware_adapter = new ArrayAdapter<String>(MainActivity.this, android.R.layout.simple_spinner_item, hardware_paths);
        hardware_adapter.setDropDownViewResource(android.R.layout.simple_spinner_dropdown_item);
        hardware_spinner.setPrompt("Choose compute backend!");
        hardware_spinner.setAdapter(hardware_adapter);

        datatype_spinner = (Spinner)findViewById(R.id.datatype_spinner);
        ArrayAdapter<String> datatype_adapter = new ArrayAdapter<String>(MainActivity.this, android.R.layout.simple_spinner_item, datatype_paths);
        datatype_adapter.setDropDownViewResource(android.R.layout.simple_spinner_dropdown_item);
        datatype_spinner.setPrompt("Choose datatype!");
        datatype_spinner.setAdapter(datatype_adapter);

        //this.selected_framework = framework_spinner.getSelectedItem().toString();
        //this.selected_model = model_spinner.getSelectedItem().toString();
        //this.selected_hardware = hardware_spinner.getSelectedItem().toString();
        // DEBUG
        //System.out.println("OnCreate: User wants to run "+this.selected_model+" deployed in "+this.selected_framework+" on mobile "+this.selected_hardware);

        framework_spinner.setOnItemSelectedListener(this);
        model_spinner.setOnItemSelectedListener(this);
        hardware_spinner.setOnItemSelectedListener(this);
        datatype_spinner.setOnItemSelectedListener(this);
    }

    @Override
    public void onItemSelected(AdapterView<?> parent, View v, int pos, long id) {

        if(parent.getId() == R.id.framework_spinner) {
            // set framework choice
            this.selected_framework = framework_spinner.getSelectedItem().toString();
        } else if(parent.getId() == R.id.model_spinner) {
            // set model choice
            this.selected_model = model_spinner.getSelectedItem().toString();
        } else if(parent.getId() == R.id.hardware_spinner) {
            // set hardware choice
            this.selected_hardware = hardware_spinner.getSelectedItem().toString();
        } else if(parent.getId() == R.id.datatype_spinner) {
            // set datatype choice
            this.selected_datatype = datatype_spinner.getSelectedItem().toString();
        }

        // DEBUG
        System.out.println("OnItemSelected: User wants to run "+this.selected_model+" deployed in "+this.selected_framework+" on mobile "+this.selected_hardware+" with datatype " + this.selected_datatype);
    }

    @Override
    public void onNothingSelected(AdapterView<?> parent) {

        if(parent.getId() == R.id.framework_spinner) {
            // default
        } else if(parent.getId() == R.id.model_spinner) {
            // default
        } else {
            // default
        }
    }

    /**
     * Predict
     * @param view -- the view that is clicked
     */
    public void goToPredict(View view){
        Intent intent = new Intent(this, CameraActivity.class);
        intent.putExtra("framework", this.selected_framework);
        intent.putExtra("model", this.selected_model);
        intent.putExtra("hardware", this.selected_hardware);
        intent.putExtra("datatype", this.selected_datatype);
        startActivity(intent);
    }

    public void toastMsg(String msg) {

        Toast toast = Toast.makeText(this, msg, Toast.LENGTH_LONG);
        toast.show();

    }

    /**
     * Deploy
     * @param view -- the view that is clicked
     */
    public void goToDeploy(View view){
        toastMsg("Deploy agent on the web (to be added)!");
    }

}
