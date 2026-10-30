# ESP32 + MPU9250 integration (NOT YET IMPLEMENTED OR TESTED)
Planned message (one JSON line per sample over Wi-Fi/UDP or serial): `{"t": <seconds>, "ax":g, "ay":g, "az":g, "gx":rad/s, "gy":rad/s, "gz":rad/s}`.
Units: accelerometer in g, gyroscope in rad/s, 50 Hz target. Add an adapter that feeds `helios.core.validate()` and `Helios.infer()`. No COM port is assumed. Decision output is advisory and must stay separate from motor code.
