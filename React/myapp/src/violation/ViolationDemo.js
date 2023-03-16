import ReactTable from "./table/ReactTable";
import { useState, useEffect } from "react";
import ImgButton from "./table/ImgButton";
import "./ViolationDemo.css"
//import data from "../test.json"


/* Data generator */
// function usersGererator(size) {
//   let items = [];
//   for (let i = 0; i < size; i++) {
//     items.push({ id: i + 1, name: `Name ${i + 1}`, age: 18 + i });
//   }
//   return items;
// }

/* Parameter */
// const tableData = usersGererator(100);




const sizePerPage = 10;

const TableHeader = () => {
  return (
    <tr>
      <th>
        編號
      </th>
      <th>
       日期
      </th>
      <th>
        車號
      </th>
      <th>
        圖片
      </th>

    </tr>
  );
};

const tableBody = (value, index) => {
  return (
    <tr key={index}>
      <td>{value.id}</td>
      <td>{value.reportDate}</td>
      <td>{value.plateNumber}</td>
      <td>
        <ImgButton imgPath={value.imgPath} />
      </td>

    </tr>
  );
};
//console.log(data);

export default function ViolationDemo() {
//  const [tableData, setTableData] = useState(data);

  const [tableData, setTableData] = useState([]);
  const [tableData2, setTableData2] = useState([]);

  //fetch lprtable
  useEffect(() => {
    (async () => {
      const data = await fetch("http://localhost:8080/lpr/getAllCars");
      const res = await data.json();
       setTableData(res);
      
    })();
  }, []);

  //fetch alprtable
  useEffect(() => {
    (async () => {
      const data = await fetch("http://localhost:8080/alpr/getAllCars");
      const res = await data.json();
       setTableData2(res);
      
    })();
  }, []);


  //如果輸入為空白
  const reload=async()=>{
    //fetch lprtable
        const data = await fetch("http://localhost:8080/lpr/getAllCars");
        const res = await data.json();
         setTableData(res);

    //fetch alprtable
    
        const data2 = await fetch("http://localhost:8080/alpr/getAllCars");
        const res2 = await data2.json();
         setTableData2(res2);
  }




  //車號查詢
  const searchCar=async()=> {
    //const postData =  parseInt(document.getElementById("plate-number").value);
    const postData = document.getElementById("plate-number").value;
    const lprurl=`http://localhost:8080/lpr/getAllSameCars/${postData}`
    const alprurl=`http://localhost:8080/alpr/getAllSameCars/${postData}`
    console.log(postData);
    if(postData===""){
      reload();
    }else{

    //search lpr table
    try {
    const data = await fetch(lprurl);
    const res = await data.json();
    console.log(res)
    setTableData(res);
    }catch (err) {
      console.error(err);
    }



    //search alpr table
    try {
      const data = await fetch(alprurl);
      const res2 = await data.json();
      console.log(res2)
      setTableData2(res2);
      }catch (err) {
        console.error(err);
      }
    }
  };



  return (
    <>
    <input type="text" id="plate-number"></input>
    <button onClick={searchCar}>車號查詢</button>
    <div className="wrapper">
      <div className="left">
        <ReactTable
          tableData={tableData}
          sizePerPage={sizePerPage}
          tableHeader={TableHeader}
          tableBody={tableBody}
        />
      </div>
      <div className="right">
          <ReactTable
          tableData={tableData2}
          sizePerPage={sizePerPage}
          tableHeader={TableHeader}
          tableBody={tableBody}
        />
      </div>
    </div>

    </>
  );
}
