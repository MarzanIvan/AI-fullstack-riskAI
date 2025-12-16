<?php
$mysqli = new mysqli("db", "admin_user", "my_password", "ParfumTim");

if ($mysqli->connect_errno) {
    die("Ошибка подключения к БД: " . $mysqli->connect_error);
}
?>