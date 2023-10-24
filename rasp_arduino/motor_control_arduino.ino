char cmd;
#include <Servo.h>
Servo ssv1;
Servo ssv2;
Servo ssv3;
Servo ssv4;

int ssv1pin = 6;
int ssv2pin = 10;
int ssv3pin = 8;
int ssv4pin = 9;

int angle = 0;


void setup() {

  // 시리얼 통신 시작 (boadrate: 9600)
  Serial.begin(9600);

  ssv1.attach(ssv1pin); //왼쪽 전
  ssv1.write(40);

  ssv2.attach(ssv2pin);//오른쪽 뒤
  ssv2.write(20);

  ssv3.attach(ssv3pin); //왼쪽 후
  ssv3.write(41);

  ssv4.attach(ssv4pin); //오른쪽 앞
  ssv4.write(0);
}

void loop() {
  delay(500);
  

  if (Serial.available()) {
//    delay(500);
    cmd = Serial.read();
//    Serial.println(cmd);

    if (cmd == 'a') {
      Serial.println("welcome");
        ssv1.write(-300);
        ssv2.write(80);
        ssv3.write(-300);
        ssv4.write(120);
        delay(5000);
      
      cmd = 'n';
    }
  
  
    else if (cmd != 'a') {
      Serial.println("stranger");
        ssv1.write(40);
        ssv2.write(20);
        ssv3.write(41);
        ssv4.write(0);
        delay(100);
      cmd = 'n';
      
    }
        
        ssv1.write(40);
        ssv2.write(20);
        ssv3.write(41);
        ssv4.write(0);

  }

}
