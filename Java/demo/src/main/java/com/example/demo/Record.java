package com.example.demo;


import java.time.LocalDateTime;

import javax.persistence.Entity;
import javax.persistence.GeneratedValue;
import javax.persistence.GenerationType;
import javax.persistence.Id;
import javax.persistence.Table;

@Entity
@Table(name="lprtable")
public class Record {
  @Id
  @GeneratedValue(strategy = GenerationType.IDENTITY)
  private Integer id;
  private String plateNumber = "";
  private String recognitionTimeStr = "";
  private String imagePath = "";
  private String cameraId = "";



  public String getCameraId() {
    return this.cameraId;
  }

  public void setCameraId(String cameraId) {
    this.cameraId = cameraId;
  }

  public String getImagePath() {
    return this.imagePath;
  }

  public void setImagePath(String imagePath) {
    this.imagePath = imagePath;
  }

  public String getRecognitionTimeStr() {
    return this.recognitionTimeStr;
  }

  public void setRecognitionTimeStr(String recognitionTimeStr) {
    this.recognitionTimeStr = recognitionTimeStr;
  }

  public Integer getId() {
    return this.id;
  }

  public void setId(Integer id) {
    this.id = id;
  }

  public String getPlateNumber() {
    return this.plateNumber;
  }

  public void setPlateNumber(String plateNumber) {
    this.plateNumber = plateNumber;
  }




  @Override
  public String toString() {
    return "{" +
      " id='" + getId() + "'" +
      ", plateNumber='" + getPlateNumber() + "'" +
      ", recognitionTimeStr='" + getRecognitionTimeStr() + "'" +
      ", imagePath='" + getImagePath() + "'" +
      ", cameraId='" + getCameraId() + "'" +
      "}";
  }
}
