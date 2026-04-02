#include <Servo.h>

Servo Base;
Servo Vertical;

int baseAngle     = 0;
int verticalAngle = 0;

void setup() {
  Base.attach(9);
  Vertical.attach(11);
  Serial.begin(9600);

  Base.write(baseAngle);
  Vertical.write(verticalAngle);
}

void loop() {
  if (Serial.available() > 0) {
    String input = Serial.readStringUntil('\n');
    input.trim();

    int commaIndex = input.indexOf(',');
    if (commaIndex != -1) {
      baseAngle     = input.substring(0, commaIndex).toInt();
      verticalAngle = input.substring(commaIndex + 1).toInt();

      baseAngle     = constrain(baseAngle,     0, 180);
      verticalAngle = constrain(verticalAngle, 0, 180);

      Base.write(baseAngle);
      Vertical.write(verticalAngle);

      Serial.print("Base: ");
      Serial.print(baseAngle);
      Serial.print(" | Vertical: ");
      Serial.println(verticalAngle);
    }
  }
}
