package com.example.android.mlmodelscopepredictor;

import android.app.Activity;
import android.content.Intent;
import android.support.v7.app.AppCompatActivity;
import android.os.Bundle;
import android.view.View;
import android.widget.AdapterView;
import android.widget.ArrayAdapter;
import android.widget.Spinner;

public class MainActivity extends Activity implements AdapterView.OnItemSelectedListener {

    private Spinner framework_spinner;
    private Spinner model_spinner;
    private Spinner hardware_spinner;

    private static final String[] framework_paths = {"TFLite", "Caffe2", "TVM"};
    private static final String[] model_paths = {"Mobilenet_1", "Mobilenet_2", "Mobilenet_3"};
    private static final String[] hardware_paths = {"CPU", "GPU", "DSP"};

    public String selected_framework;
    public String selected_model;
    public String selected_hardware;

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

        this.selected_framework = framework_spinner.getSelectedItem().toString();
        this.selected_model = model_spinner.getSelectedItem().toString();
        this.selected_hardware = hardware_spinner.getSelectedItem().toString();
        System.out.println("User wants to run "+this.selected_model+" deployed in "+this.selected_framework+" on mobile "+this.selected_hardware);
        //framework_spinner.setOnItemSelectedListener(this);
        //model_spinner.setOnItemSelectedListener(this);
        //hardware_spinner.setOnItemSelectedListener(this);
    }

    @Override
    public void onItemSelected(AdapterView<?> parent, View v, int pos, long id) {

        if(parent.getId() == R.id.framework_spinner) {
            // set framework choice
        } else if(parent.getId() == R.id.model_spinner) {
            // set model choice
        } else {
            // set hardware choice
        }
    }

    @Override
    public void onNothingSelected(AdapterView<?> parent) {

        if(parent.getId() == R.id.framework_spinner) {
            // force user to select framework
        } else if(parent.getId() == R.id.model_spinner) {
            // force user to select model
        } else {
            // force user to select hardware
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
        startActivity(intent);
    }

}
