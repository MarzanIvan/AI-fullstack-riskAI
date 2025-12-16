DELIMITER $$

CREATE PROCEDURE remove_dataset_from_model (
    IN p_model_name VARCHAR(255),
    IN p_dataset_name VARCHAR(255)
)
BEGIN
    DECLARE v_model_id BIGINT UNSIGNED DEFAULT NULL;
    DECLARE v_dataset_id BIGINT UNSIGNED DEFAULT NULL;
    DECLARE v_count BIGINT DEFAULT 0;

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
    LIMIT 1;

    IF v_dataset_id IS NULL THEN
        SIGNAL SQLSTATE '45000'
            SET MESSAGE_TEXT = 'Dataset not found';
    END IF;

    DELETE FROM model_datasets
    WHERE model_id = v_model_id
      AND dataset_id = v_dataset_id;

    SELECT COUNT(*) INTO v_count
    FROM model_datasets
    WHERE dataset_id = v_dataset_id;

    IF v_count = 0 THEN
        DELETE FROM datasets
        WHERE id = v_dataset_id;
    END IF;

END$$

DELIMITER ;
