<?php
session_start();
header('Content-Type: application/json; charset=utf-8');

if (!isset($_SESSION['auth']) || $_SESSION['auth'] !== true) {
    echo json_encode(['success'=>false,'message'=>'Не авторизован']);
    exit();
}

$model_name = $_POST['model_name'] ?? '';
$model_path = $_POST['model_path'] ?? '';

if(!$model_name || !$model_path){
    echo json_encode(['success'=>false,'message'=>'Все поля обязательны']);
    exit();
}

$mysqli = new mysqli("mysql_db", "root", "root", "riskai");
if ($mysqli->connect_errno) {
    echo json_encode(['success'=>false,'message'=>'Ошибка подключения к БД']);
    exit();
}

// Вставляем модель
$stmt = $mysqli->prepare("INSERT INTO models (name, path) VALUES (?, ?)");
$stmt->bind_param("ss", $model_name, $model_path);

if($stmt->execute()){
    echo json_encode(['success'=>true,'message'=>'Модель добавлена']);
} else {
    echo json_encode(['success'=>false,'message'=>'Ошибка при добавлении модели: '.$stmt->error]);
}

$stmt->close();
$mysqli->close();
