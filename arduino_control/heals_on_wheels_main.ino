#include <Arduino.h>
#include <Wire.h>
#include <TFLI2C.h>
#include <Servo.h>

TFLI2C tflI2C;

int16_t tfDist;
int16_t tfAddr = TFL_DEF_ADR;

int motor1pin1 = 2, motor1pin2 = 3, motor2pin1 = 4, motor2pin2 = 5;
const int trigPinLeft = 7, echoPinLeft = 8, trigPinRight = 12, echoPinRight = 13;
float durationLeft, durationRight, distanceLeft, distanceRight;

Servo myservo;
bool servoDirection = true;
unsigned long previousServoMillis = 0, servoInterval = 190;
const float obstacleThreshold = 30.0; 

// Configuration
void setup() {

  for (int pin : {motor1pin1, motor1pin2, motor2pin1, motor2pin2, 11, 10, trigPinLeft, trigPinRight})
    pinMode(pin, OUTPUT);
  
  pinMode(echoPinLeft, INPUT);
  pinMode(echoPinRight, INPUT);
  
  myservo.attach(9);
  Serial.begin(115200);
  Wire.begin();
}

void loop() {
  unsigned long currentMillis = millis();
  // Primary drive logic from lidar
  if (tflI2C.getData(tfDist, tfAddr)) {
    if (tfDist > 0) {
      digitalWrite(motor1pin1, LOW);
      digitalWrite(motor1pin2, HIGH);
      digitalWrite(motor2pin1, HIGH);
      digitalWrite(motor2pin2, LOW);
      analogWrite(11, 255);
      analogWrite(10, 255);
    } else {
      analogWrite(11, 0);
      analogWrite(10, 0);
    }
  }

  // Trigger ultrasonic sensors
  for (int pin : {trigPinLeft, trigPinRight}) {
    digitalWrite(pin, LOW);
    delayMicroseconds(2);
    digitalWrite(pin, HIGH);
    delayMicroseconds(10);
    digitalWrite(pin, LOW);
  }
  // Calculate range values from ultrasonic sensors
  durationLeft = pulseIn(echoPinLeft, HIGH);
  durationRight = pulseIn(echoPinRight, HIGH);
  distanceLeft = durationLeft * 0.01715;
  distanceRight = durationRight * 0.01715;

  // Decision making based on sensor distances
  if (distanceLeft <= obstacleThreshold && distanceRight <= obstacleThreshold) {
    // Stop if both sensors detect an obstacle
    analogWrite(11, 0);
    analogWrite(10, 0);
  } else if (distanceLeft <= obstacleThreshold) {
    // Turn right if left sensor detects an obstacle
    digitalWrite(motor1pin1, HIGH);
    digitalWrite(motor1pin2, LOW);
    digitalWrite(motor2pin1, LOW);
    digitalWrite(motor2pin2, HIGH);
    analogWrite(11, 255);
    analogWrite(10, 255);
  } else if (distanceRight <= obstacleThreshold) {
    // Turn left if right sensor detects an obstacle
    digitalWrite(motor1pin1, LOW);
    digitalWrite(motor1pin2, HIGH);
    digitalWrite(motor2pin1, HIGH);
    digitalWrite(motor2pin2, LOW);
    analogWrite(11, 255);
    analogWrite(10, 255);
  } else {
    // Move forward if no obstacles
    digitalWrite(motor1pin1, LOW);
    digitalWrite(motor1pin2, HIGH);
    digitalWrite(motor2pin1, HIGH);
    digitalWrite(motor2pin2, LOW);
    analogWrite(11, 255);
    analogWrite(10, 255);
  }
  // Delay to avoid overloading
  delay(50);
}
