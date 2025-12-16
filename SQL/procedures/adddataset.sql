DELIMITER $$

CREATE PROCEDURE add_dataset_to_model (
    IN p_model_name VARCHAR(255),
    IN p_dataset_name VARCHAR(255),
    IN p_dataset_path VARCHAR(512)
)
BEGIN
    DECLARE v_model_id BIGINT UNSIGNED;
    DECLARE v_dataset_id BIGINT UNSIGNED;

    SELECT id
    INTO v_model_id
    FROM models
    WHERE name = p_model_name
    LIMIT 1;

    IF v_model_id IS NULL THEN
        SIGNAL SQLSTATE '45000'
        SET MESSAGE_TEXT = 'Model not found';
    END IF;

    SELECT id
    INTO v_dataset_id
    FROM datasets
    WHERE name = p_dataset_name
       OR path = p_dataset_path
    LIMIT 1;

    IF v_dataset_id IS NULL THEN
        INSERT INTO datasets (name, path)
        VALUES (p_dataset_name, p_dataset_path);

        SET v_dataset_id = LAST_INSERT_ID();
    END IF;

    INSERT IGNORE INTO model_datasets (model_id, dataset_id)
    VALUES (v_model_id, v_dataset_id);

END$$

DELIMITER ;
