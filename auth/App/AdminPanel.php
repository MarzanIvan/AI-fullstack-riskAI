<?php
session_start();
if (!isset($_SESSION['auth']) || $_SESSION['auth'] !== true) {
    header("Location: AuthAdmin.php");
    exit();
}

// Подключение к MySQL
$mysqli = new mysqli("mysql_db", "root", "root", "riskai");
if ($mysqli->connect_errno) {
    die("Ошибка подключения: " . $mysqli->connect_error);
}

// Получаем все модели
$models_result = $mysqli->query("SELECT id, name, path FROM models ORDER BY name");

// Получаем все датасеты
$datasets_result = $mysqli->query("SELECT d.id, d.name, d.path, m.name AS model_name
    FROM datasets d
    LEFT JOIN model_datasets md ON md.dataset_id = d.id
    LEFT JOIN models m ON md.model_id = m.id
    ORDER BY d.id");

// Для формы выбора модели при добавлении датасета
$models_select = $mysqli->query("SELECT name FROM models ORDER BY name");
?>

<!DOCTYPE html>
<html lang="ru">
<head>
<meta charset="UTF-8">
<title>Admin Panel</title>
<style>
body { font-family: Arial; padding: 20px; }
h1 { margin-bottom: 20px; }
table { border-collapse: collapse; width: 100%; margin-bottom: 30px; }
th, td { border: 1px solid #ccc; padding: 8px; text-align: left; }
button { padding: 8px 16px; margin-top: 10px; cursor: pointer; }
form { margin-bottom: 30px; }
input, select { padding: 8px; margin: 4px 0; }
</style>
</head>
<body>

<h1>Модели</h1>
<table>
<tr><th>ID</th><th>Название</th><th>Путь</th></tr>
<?php while ($row = $models_result->fetch_assoc()): ?>
<tr>
    <td><?= $row['id'] ?></td>
    <td><?= $row['name'] ?></td>
    <td><?= $row['path'] ?></td>
</tr>
<?php endwhile; ?>
</table>

<h2>Добавить модель</h2>
<form id="addModelForm">
    <input type="text" name="model_name" placeholder="Название модели" required>
    <input type="text" name="model_path" placeholder="Путь к модели" required>
    <button type="submit">Добавить модель</button>
</form>

<h2>Добавить датасет</h2>
<form id="addDatasetForm">
    <input type="text" name="dataset_name" placeholder="Название датасета" required>
    <input type="text" name="dataset_path" placeholder="Путь к датасету" required>
    <select name="model_name" required>
        <option value="">Выберите модель</option>
        <?php while ($row = $models_select->fetch_assoc()): ?>
            <option value="<?= $row['name'] ?>"><?= $row['name'] ?></option>
        <?php endwhile; ?>
    </select>
    <button type="submit">Добавить датасет</button>
</form>

<h2>Существующие датасеты</h2>
<table id="datasetsTable">
<tr><th>ID</th><th>Название</th><th>Путь</th><th>Модель</th><th>Обучить</th></tr>
<?php while ($row = $datasets_result->fetch_assoc()): ?>
<tr>
    <td><?= $row['id'] ?></td>
    <td><?= $row['name'] ?></td>
    <td><?= $row['path'] ?></td>
    <td><?= $row['model_name'] ?? '-' ?></td>
    <td>
	<button onclick="trainModel(
    '<?= $row['model_name'] ?>',
    '<?= $row['name'] ?>',
    '<?= $row['path'] ?>'
)">
    Обучить
</button>
    </td>
</tr>
<?php endwhile; ?>
</table>

<script>
// Добавление модели через AJAX
document.getElementById("addModelForm").addEventListener("submit", async (e) => {
    e.preventDefault();
    const formData = new FormData(e.target);
    const data = new URLSearchParams(formData).toString();

    const resp = await fetch("/php_api/AddModel.php", {
        method: "POST",
        headers: { "Content-Type": "application/x-www-form-urlencoded" },
        body: data
    });
    const result = await resp.json();
    alert(result.message);
    if(result.success) location.reload();
});

// Добавление датасета через процедуру
document.getElementById("addDatasetForm").addEventListener("submit", async (e) => {
    e.preventDefault();
    const formData = new FormData(e.target);
    const dataset_name = formData.get("dataset_name");
    const dataset_path = formData.get("dataset_path");
    const model_name = formData.get("model_name");

    const resp = await fetch("/php_api/AddDataset.php", {
        method: "POST",
        headers: { "Content-Type": "application/x-www-form-urlencoded" },
        body: `p_model_name=${encodeURIComponent(model_name)}&p_dataset_name=${encodeURIComponent(dataset_name)}&p_dataset_path=${encodeURIComponent(dataset_path)}`
    });
    const result = await resp.json();
    alert(result.message);
    if(result.success) location.reload();
});

async function trainModel(modelName, datasetName, datasetPath) {
    let route = '';
    if (modelName.toLowerCase().includes('credit')) route = '/api/ml_credit';
    else if (modelName.toLowerCase().includes('investment')) route = '/api/ml_investment';
    else if (modelName.toLowerCase().includes('insurance')) route = '/api/ml_insurance';
    else {
        alert('Неизвестная модель');
        return;
    }

    try {
        const response = await fetch(route, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                dataset_name: datasetName,
                dataset_path: datasetPath
            })
        });

        const result = await response.json();
        alert("Обучение завершено:\n" + JSON.stringify(result, null, 2));
    } catch (err) {
        alert("Ошибка при обучении: " + err);
    }
}
</script>

</body>
</html>
