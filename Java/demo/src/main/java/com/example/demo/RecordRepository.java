package com.example.demo;

import java.util.ArrayList;
import java.util.Optional;

import org.springframework.data.jpa.repository.Query;
import org.springframework.data.repository.CrudRepository;

public interface RecordRepository extends CrudRepository<Record, Integer> {



  Iterable<Record> findAllByOrderByIdDesc();

    //search車號
    @Query(value = "SELECT * FROM lprtable WHERE plate_number=?1 ORDER by id DESC", nativeQuery = true)
    ArrayList<Record> findByPlateNumber(String plateNumber);
    // //search車號
    // @Query(value = "SELECT * FROM alprtable WHERE plate_number LIKE ?1 ORDER by id DESC LIMIT 0,1", nativeQuery = true)
    // ArrayList<Record> findByPlateNumber(String plateNumber);


  @Query(value = "SELECT * FROM lprtable WHERE camera_id=?1 ORDER by id DESC LIMIT 0,1", nativeQuery = true)
  Optional<Record> findAllLatestRecordByCameraId(String cameraId);

  @Query(value = "SELECT * FROM lprtable WHERE plate_number LIKE ?1 AND name LIKE ?2 AND vehicle_type LIKE ?3 ", nativeQuery = true)
  Iterable<Record> searchWithoutDate(String plateNumber, String name, String vehicleType);

  @Query(value = "SELECT * FROM lprtable WHERE plate_number LIKE ?1 AND name LIKE ?2 AND vehicle_type LIKE ?3 AND (DATE(recognition_time) BETWEEN ?4 AND ?5)", nativeQuery = true)
  Iterable<Record> searchByDateBetween(String plateNumber, String name, String vehicleType, String startDate,
      String endDate);
}
