/*
  Combine H (2D) linear guide and another 1D linear guide for synchronous passthrough data collection. 

  Vertical rail: 2 stepper motors (left and right) for better stability, driven by same step signal and direction signal.
  Horizontal rail: 1 stepper motor.
  Trigger: 1 GPIO pin for triggering raspi which will trigger multiple radars through hardware trigger.
*/

#include <AccelStepper.h>
#include <math.h>

// for H rail
#define PUL_Ver_L 3
#define DIR_Ver_L 8
#define PUL_Ver_R 4
#define DIR_Ver_R 9
#define PUL_Hor 5
#define DIR_Hor 10
#define TRIGGER 2

// for 1D rail
#define PUL_1D 6
#define DIR_1D 11

// used as 5V pulled up
#define PULL_UP_0 12
#define PULL_UP_1 13


byte pul_ver_l = B00000001 << PUL_Ver_L;
byte pul_ver_r = B00000001 << PUL_Ver_R;
byte pul_hor = B00000001 << PUL_Hor;
byte trigger = B00000001 << TRIGGER;

byte pul_1d = B00000001 << PUL_1D;


long RATIO = 200;
// float speed = 23; // 50 -> about 40mm/s, // 23 -> 20mm/s
float speed = 6;
unsigned int pd = 1e6/speed/RATIO/2;


void setup() {
  Serial.begin(9600);
  Serial.setTimeout(10);

  DDRD = pul_ver_l|pul_ver_r|pul_hor|trigger|pul_1d;
  PORTD = B00000000;

  // pinMode(PUL_Ver_L, OUTPUT);
  pinMode(DIR_Ver_L, OUTPUT);
  // pinMode(PUL_Ver_R, OUTPUT);
  pinMode(DIR_Ver_R, OUTPUT);
  // pinMode(PUL_Hor, OUTPUT);
  pinMode(DIR_Hor, OUTPUT);
  // pinMode(TRIGGER, OUTPUT);
  pinMode(DIR_1D, OUTPUT);

  // pull up
  pinMode(PULL_UP_0, OUTPUT);
  pinMode(PULL_UP_1, OUTPUT);
  digitalWrite(PULL_UP_0, HIGH);
  digitalWrite(PULL_UP_1, HIGH);

}


void loop() {

  // Serial.println("Enter x,y:");
  while (Serial.available() == 0) {}     //wait for data available
  String teststr = Serial.readString();  //read until timeout
  teststr.trim();                        // remove any \r \n whitespace at the end of the String

  // input string will has length 4 each representing: H_x, H_y, 1D_x, step_to_trigger

  char *token = strtok(teststr.c_str(), ",");
  double cmdAry[4];
  int i=0;
  while( token != NULL ) {
      // cmdAry[i++] = atoi(token);
      cmdAry[i++] = atof(token);
      // cmdAry[i++] = strtol(token);
      token = strtok(NULL, ",");
  }
  
  double posx = cmdAry[0];
  double posy = cmdAry[1];
  double pos1d = cmdAry[2];
  double num_to_trigger;

  if (i > 3){
    num_to_trigger = cmdAry[3];
    pulse_train_trigger_byte(posx, posy, pos1d, int(round(num_to_trigger)));
  }else{
    pulse_train_byte(posx, posy, pos1d);
  }

  Serial.println("Done.");

}


void pulse_train_trigger_byte(double num_x, double num_y, double num_1d, int num_to_trigger) {

  num_x = long(round(num_x));
  num_y = long(round(num_y));
  num_1d = long(round(num_1d));
  // num_to_trigger = int(round(num_to_trigger));

  if (num_x > 0){
    digitalWrite(DIR_Hor, LOW);
  } else{
    digitalWrite(DIR_Hor, HIGH);
    num_x = -num_x;
  }
  if (num_y > 0){
    digitalWrite(DIR_Ver_L, LOW);
    digitalWrite(DIR_Ver_R, LOW);
  } else{
    digitalWrite(DIR_Ver_L, HIGH);
    digitalWrite(DIR_Ver_R, HIGH);
    num_y = -num_y;
  }

  if (num_1d > 0){
    digitalWrite(DIR_1D, LOW);
  } else{
    digitalWrite(DIR_1D, HIGH);
    num_1d = -num_1d;
  }

  long s = 0;

  // first will be in-place trigger
  PORTD = trigger; 
  delayMicroseconds(pd);
  PORTD = B00000000;
  delayMicroseconds(pd);

  while (s < num_x || s < num_y || s < num_1d){

    byte gpio_enable = B00000000;
    if (s<num_x) 
      gpio_enable |= pul_hor;
    if (s<num_y) 
      gpio_enable |= pul_ver_l | pul_ver_r;
    if (s%num_to_trigger == 0)
      gpio_enable |= trigger;
    
    if (s<num_1d) 
      gpio_enable |= pul_1d;

    PORTD = gpio_enable;

    delayMicroseconds(pd);
    
    PORTD = B00000000;

    delayMicroseconds(pd);

    s++;

  }
  // supplement triggering
  if (s%num_to_trigger == 0 && s!=0){
    // first will be in-place trigger
    PORTD = trigger; 
    delayMicroseconds(pd);
    PORTD = B00000000;
    delayMicroseconds(pd);
  }

}


void pulse_train_byte(double num_x, double num_y, double num_1d) {

  num_x = long(round(num_x));
  num_y = long(round(num_y));
  num_1d = long(round(num_1d));
  // num_to_trigger = int(round(num_to_trigger));

  if (num_x > 0){
    digitalWrite(DIR_Hor, LOW);
  } else{
    digitalWrite(DIR_Hor, HIGH);
    num_x = -num_x;
  }
  if (num_y > 0){
    digitalWrite(DIR_Ver_L, LOW);
    digitalWrite(DIR_Ver_R, LOW);
  } else{
    digitalWrite(DIR_Ver_L, HIGH);
    digitalWrite(DIR_Ver_R, HIGH);
    num_y = -num_y;
  }

  if (num_1d > 0){
    digitalWrite(DIR_1D, LOW);
  } else{
    digitalWrite(DIR_1D, HIGH);
    num_1d = -num_1d;
  }

  long s = 0;

  while (s < num_x || s < num_y || s < num_1d){

    byte gpio_enable = B00000000;
    if (s<num_x) 
      gpio_enable |= pul_hor;
    if (s<num_y) 
      gpio_enable |= pul_ver_l | pul_ver_r;
    // if (s%2 == 0) {pul_ver_l | pul_ver_r;} // for same speed as trigger function
  
    if (s<num_1d) 
      gpio_enable |= pul_1d;

    PORTD = gpio_enable;    

    delayMicroseconds(pd);
    
    PORTD = B00000000;

    delayMicroseconds(pd);

    s++;

  }
}


void pulse_train_trigger(double num_x, double num_y, int num_to_trigger) {

  num_x = long(round(num_x));
  num_y = long(round(num_y));
  // num_to_trigger = int(round(num_to_trigger));

  if (num_x > 0){
    digitalWrite(DIR_Hor, LOW);
  } else{
    digitalWrite(DIR_Hor, HIGH);
    num_x = -num_x;
  }
  if (num_y > 0){
    digitalWrite(DIR_Ver_L, LOW);
    digitalWrite(DIR_Ver_R, LOW);
  } else{
    digitalWrite(DIR_Ver_L, HIGH);
    digitalWrite(DIR_Ver_R, HIGH);
    num_y = -num_y;
  }

  long s = 0;

  while (s < num_x || s < num_y){

    // step motor
    if (s < num_x){
      digitalWrite(PUL_Hor, HIGH);
    }
    if (s < num_y){
      digitalWrite(PUL_Ver_L, HIGH);
      digitalWrite(PUL_Ver_R, HIGH);    
    }

    // trigger raspi
    if (s%num_to_trigger == 0) {
      digitalWrite(TRIGGER, HIGH); 
    }

    delayMicroseconds(pd);
    
    if (s < num_x){
      digitalWrite(PUL_Hor, LOW);
    }
    if (s < num_y){
      digitalWrite(PUL_Ver_L, LOW);
      digitalWrite(PUL_Ver_R, LOW);    
    }

    if (s%num_to_trigger == 0) {
      digitalWrite(TRIGGER, LOW); 
    }

    delayMicroseconds(pd);

    s++;

  }
}


void pulse_train(double num_x, double num_y) {

  num_x = long(round(num_x));
  num_y = long(round(num_y));

  if (num_x > 0){
    digitalWrite(DIR_Hor, LOW);
  } else{
    digitalWrite(DIR_Hor, HIGH);
    num_x = -num_x;
  }
  if (num_y > 0){
    digitalWrite(DIR_Ver_L, LOW);
    digitalWrite(DIR_Ver_R, LOW);
  } else{
    digitalWrite(DIR_Ver_L, HIGH);
    digitalWrite(DIR_Ver_R, HIGH);
    num_y = -num_y;
  }

  // Serial.println(num_x);
  // Serial.println(num_y);
  // Serial.println(STEPS);
  // Serial.println(s);
  long s_x = 0;
  long s_y = 0;

  while (s_x < num_x || s_y < num_y){
    if (s_x < num_x){
      digitalWrite(PUL_Hor, HIGH);
    }
    if (s_y < num_y){
      digitalWrite(PUL_Ver_L, HIGH);
      digitalWrite(PUL_Ver_R, HIGH);    
    }

    delayMicroseconds(pd);
    
    if (s_x < num_x){
      digitalWrite(PUL_Hor, LOW);
    }
    if (s_y < num_y){
      digitalWrite(PUL_Ver_L, LOW);
      digitalWrite(PUL_Ver_R, LOW);    
    }

    delayMicroseconds(pd);

    s_x++;
    s_y++;
  }
}


void printBin(byte aByte) {
  for (int8_t aBit = 7; aBit >= 0; aBit--)
    Serial.write(bitRead(aByte, aBit) ? '1' : '0');
}
