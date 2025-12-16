<?php
session_start();
header('Content-Type: application/json; charset=utf-8');

if (!isset($_SESSION['auth']) || $_SESSION['auth'] !== true) {
    echo json_encode(['success'=>false,'message'=>'Не авторизован']);
    exit();
}

$p_model_name = $_POST['p_model_name'] ?? '';
$p_dataset_name = $_POST['p_dataset_name'] ?? '';
$p_dataset_path = $_POST['p_dataset_path'] ?? '';

if(!$p_model_name || !$p_dataset_name || !$p_dataset_path){
    echo json_encode(['success'=>false,'message'=>'Все поля обязательны']);
    exit();
}

$mysqli = new mysqli("mysql_db", "root", "root", "riskai");
if ($mysqli->connect_errno) {
    echo json_encode(['success'=>false,'message'=>'Ошибка подключения к БД']);
    exit();
}

$sql = "CALL add_dataset_to_model(?, ?, ?)";
$stmt = $mysqli->prepare($sql);
$stmt->bind_param("sss", $p_model_name, $p_dataset_name, $p_dataset_path);

try {
    $stmt->execute();
    echo json_encode(['success'=>true,'message'=>'Датасет успешно добавлен']);
} catch (Exception $e) {
    echo json_encode(['success'=>false,'message'=>$stmt->error]);
}

$stmt->close();
$mysqli->close();
