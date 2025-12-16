<?php
session_start();

// Разрешаем CORS для разработки (если нужно)
header('Content-Type: application/json; charset=utf-8');

if ($_SERVER['REQUEST_METHOD'] !== 'POST') {
    echo json_encode(['success' => false, 'message' => 'Метод запроса должен быть POST']);
    exit();
}

$password = $_POST['password'] ?? '';
$correct_hash = '$2y$10$rp/wovQfllua/PXVErlrOOdlVIJCfLl4IEL7OpGwNBgkO5Pdf6HiS';

if (password_verify($password, $correct_hash)) {
    $_SESSION['auth'] = true;
    echo json_encode(['success' => true, 'message' => 'Авторизация успешна']);
} else {
    echo json_encode(['success' => false, 'message' => 'Неверный пароль']);
}
