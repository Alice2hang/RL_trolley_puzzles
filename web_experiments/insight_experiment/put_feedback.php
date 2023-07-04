<?php

// this path should point to your configuration file.
include('feedback_config.php');

$data_array = json_decode(file_get_contents('php://input'), true);

try {
  $conn = new PDO("mysql:host=$servername;port=$port;dbname=$dbname", $username, $password);
  $conn->setAttribute(PDO::ATTR_ERRMODE, PDO::ERRMODE_EXCEPTION);
  // First stage is to get all column names from the table and store
  // them in $col_names array.
  $stmt = $conn->prepare("SHOW COLUMNS FROM `$table`");
  $stmt->execute();
  $col_names = array();

  while($row = $stmt->fetchColumn()) {
    $col_names[] = $row;
  }
  // Second stage is to create prepared SQL statement using the column
  // names as a guide to what values might be in the JSON.  
  $sql = "INSERT INTO $table VALUES(";
  for($i = 0; $i < count($col_names); $i++){
    $name = $col_names[$i];
    $sql .= ":$name";
    if($i != count($col_names)-1){
      $sql .= ", ";
    }
  }
  
  $sql .= ");";
  $insertstmt = $conn->prepare($sql);
  for($i = 0; $i < count($col_names); $i++){
    $colname = $col_names[$i];
    $insertstmt->bindValue(":$colname", $data_array[$colname]); 
  }
  $insertstmt->execute();

  echo 'true';
} catch(PDOException $e) {
  echo 'false';
}
$conn = null;
?>