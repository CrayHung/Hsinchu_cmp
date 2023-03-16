package com.example.demo;

import java.io.IOException;
import java.time.LocalDateTime;
import java.time.format.DateTimeFormatter;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Optional;

import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.http.HttpStatus;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.CrossOrigin;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.PathVariable;
import org.springframework.web.bind.annotation.PostMapping;
import org.springframework.web.bind.annotation.RequestBody;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RestController;
import org.springframework.web.server.ResponseStatusException;
import org.springframework.web.socket.TextMessage;
import org.springframework.web.socket.WebSocketSession;

import com.example.demo.*;
// import com.twoway.gamaniacms.entity.RecordRepository;
// import com.twoway.gamaniacms.handler.SocketTextHandler;
// import com.twoway.gamaniacms.model.RecordSearch;
// import com.twoway.gamaniacms.tool.GateTool;

@CrossOrigin
@RestController
@RequestMapping("/lpr")
public class LprController {


  @Autowired
  RecordRepository recordRepository;

  @GetMapping("/getAllCars")
  public ResponseEntity<List<Record>> getAllUsers(){
      return new ResponseEntity<List<Record>>(((List<Record>) recordRepository.findAllByOrderByIdDesc()) , HttpStatus.OK);
  }


    @GetMapping("/getById/{id}")
    public ResponseEntity<Record> getUserByID(@PathVariable("id") int ID){
      return new ResponseEntity<Record>(recordRepository.findById(ID).orElse(null), HttpStatus.OK);
    }


     //search車號
     @GetMapping("/getAllSameCars/{platenumber}")
     public ResponseEntity<List<Record>> getAllSamePlateNumber(@PathVariable("platenumber") String plateNumber){
         List<Record> pp = recordRepository.findByPlateNumber(plateNumber);
 
           if (!pp.isEmpty()) {
            System.out.println(pp);
           return ResponseEntity.ok(pp);
        }
        return ResponseEntity.noContent().build();
        
     }




}
