package com.example.android.mlmodelscopepredictor;

import android.app.Activity;
import android.support.v7.app.AppCompatActivity;
import android.os.Bundle;
import android.view.View;

public class MainActivity extends Activity {

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);
    }

    /**
     * Choose framework
     * @param view -- the view that is clicked
     */
    public void MLFrameworkMe(View view){
        // Framework myFramework = Framework.makeText(this, message, duration);
        //MLFramework myMLFramework = MLFramework.makeText(this, "Choose Framework",
        //        MLFramework.LENGTH_SHORT);
        //myMLFramework.show();
        // TODO
    }

    /**
     * Choose Model
     * @param view -- the view that is clicked
     */
    public void MLModelMe(View view){
        // TODO
    }

    /**
     * Choose Hardware
     * @param view -- the view that is clicked
     */
    public void MLHardwareMe(View view){
        // TODO
    }
}
