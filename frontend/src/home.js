import React, { useState, useEffect } from "react";
import axios from "axios";

import { makeStyles, withStyles } from "@material-ui/core/styles";
import AppBar from "@material-ui/core/AppBar";
import Toolbar from "@material-ui/core/Toolbar";
import Typography from "@material-ui/core/Typography";
import Avatar from "@material-ui/core/Avatar";
import Container from "@material-ui/core/Container";
import Card from "@material-ui/core/Card";
import CardContent from "@material-ui/core/CardContent";

import {
  Paper,
  CardActionArea,
  CardMedia,
  Grid,
  TableContainer,
  Table,
  TableBody,
  TableHead,
  TableRow,
  TableCell,
  Button,
  CircularProgress,
} from "@material-ui/core";

import { DropzoneArea } from "material-ui-dropzone";
import { common } from "@material-ui/core/colors";
import Clear from "@material-ui/icons/Clear";

import cblogo from "./cblogo.PNG";
import backgroundImage from "./bg.png";

const API_URL = "http://127.0.0.1:8001/predict";

const ColorButton = withStyles((theme) => ({
  root: {
    color: theme.palette.getContrastText(common.white),
    backgroundColor: common.white,
    "&:hover": {
      backgroundColor: "#ffffff7a",
    },
  },
}))(Button);

const useStyles = makeStyles((theme) => ({
  grow: {
    flexGrow: 1,
  },
  mainContainer: {
    backgroundImage: `url(${backgroundImage})`,
    backgroundRepeat: "no-repeat",
    backgroundPosition: "center",
    backgroundSize: "cover",
    minHeight: "93vh",
    marginTop: "8px",
  },
  gridContainer: {
    justifyContent: "center",
    padding: "4em 1em 0 1em",
  },
  imageCard: {
    margin: "auto",
    maxWidth: 400,
    minHeight: 400,
    backgroundColor: "transparent",
    boxShadow: "0px 9px 70px 0px rgb(0 0 0 / 30%) !important",
    borderRadius: "15px",
  },
  media: {
    height: 400,
    objectFit: "contain",
  },
  detail: {
    backgroundColor: "white",
    display: "flex",
    justifyContent: "center",
    flexDirection: "column",
    alignItems: "center",
  },
  tableContainer: {
    backgroundColor: "transparent !important",
    boxShadow: "none !important",
  },
  tableCell: {
    fontSize: "16px",
    color: "#000000a6 !important",
    fontWeight: "bold",
    overflowWrap: "anywhere",
  },
  tableCell1: {
    fontSize: "14px",
    color: "#000000a6 !important",
    fontWeight: "bold",
  },
  buttonGrid: {
    maxWidth: "416px",
    width: "100%",
  },
  clearButton: {
    width: "100%",
    borderRadius: "15px",
    padding: "15px 22px",
    color: "#000000a6",
    fontSize: "20px",
    fontWeight: 900,
  },
  appbar: {
    background: "#be6a77",
    boxShadow: "none",
    color: "white",
  },
  loader: {
    color: "#be6a77 !important",
  },
}));

export const ImageUpload = () => {
  const classes = useStyles();

  const [selectedFile, setSelectedFile] = useState(null);
  const [preview, setPreview] = useState(null);
  const [data, setData] = useState(null);
  const [isLoading, setIsLoading] = useState(false);
  const [errorMessage, setErrorMessage] = useState("");

  // Send the selected image to FastAPI
  const sendFile = async (file) => {
    if (!file) return;

    setIsLoading(true);
    setData(null);
    setErrorMessage("");

    try {
      const formData = new FormData();
      formData.append("file", file);

      const response = await axios.post(API_URL, formData);

      setData(response.data);
    } catch (error) {
      console.error("Prediction request failed:", error);

      if (error.response) {
        setErrorMessage(
          `API error (${error.response.status}): ${
            error.response.data?.detail || "Unable to process the image."
          }`
        );
      } else {
        setErrorMessage(
          "Could not connect to FastAPI. Make sure the backend is running on port 8000."
        );
      }
    } finally {
      setIsLoading(false);
    }
  };

  // Select an image and create its preview
  const onSelectFile = (files) => {
    if (!files || files.length === 0) {
      clearData();
      return;
    }

    const file = files[0];

    if (!file.type.startsWith("image/")) {
      setErrorMessage("Please select a valid image file.");
      return;
    }

    setSelectedFile(file);
    setPreview(URL.createObjectURL(file));
    setData(null);
    setErrorMessage("");

    // Predict immediately after selecting the image
    sendFile(file);
  };

  // Clean up preview URL when it changes or the component unmounts
  useEffect(() => {
    return () => {
      if (preview) {
        URL.revokeObjectURL(preview);
      }
    };
  }, [preview]);

  const clearData = () => {
    setSelectedFile(null);
    setPreview(null);
    setData(null);
    setErrorMessage("");
    setIsLoading(false);
  };

  const confidence =
    data && data.confidence !== undefined
      ? (Number(data.confidence) * 100).toFixed(2)
      : null;

  return (
    <React.Fragment>
      <AppBar position="static" className={classes.appbar}>
        <Toolbar>
          <Typography variant="h6" noWrap>
            Potato Disease Classification
          </Typography>

          <div className={classes.grow} />

          <Avatar src={cblogo} alt="Project logo" />
        </Toolbar>
      </AppBar>

      <Container
        maxWidth={false}
        className={classes.mainContainer}
        disableGutters
      >
        <Grid
          className={classes.gridContainer}
          container
          direction="row"
          justifyContent="center"
          alignItems="center"
          spacing={2}
        >
          <Grid item xs={12}>
            <Card className={classes.imageCard}>
              {!selectedFile && (
                <CardContent>
                  <DropzoneArea
                    acceptedFiles={["image/*"]}
                    dropzoneText="Drag and drop a potato leaf image or click to upload"
                    filesLimit={1}
                    onChange={onSelectFile}
                    showPreviews={false}
                    showPreviewsInDropzone={false}
                  />
                </CardContent>
              )}

              {selectedFile && preview && (
                <CardActionArea>
                  <CardMedia
                    className={classes.media}
                    image={preview}
                    component="img"
                    title="Uploaded potato leaf"
                  />
                </CardActionArea>
              )}

              {isLoading && (
                <CardContent className={classes.detail}>
                  <CircularProgress className={classes.loader} />
                  <Typography variant="h6">
                    Processing image...
                  </Typography>
                </CardContent>
              )}

              {errorMessage && (
                <CardContent>
                  <Typography color="error">
                    {errorMessage}
                  </Typography>
                </CardContent>
              )}

              {data && (
                <CardContent className={classes.detail}>
                  <TableContainer
                    component={Paper}
                    className={classes.tableContainer}
                  >
                    <Table size="small" aria-label="Prediction results">
                      <TableHead>
                        <TableRow>
                          <TableCell className={classes.tableCell1}>
                            Predicted class
                          </TableCell>
                          <TableCell
                            align="right"
                            className={classes.tableCell1}
                          >
                            Confidence
                          </TableCell>
                        </TableRow>
                      </TableHead>

                      <TableBody>
                        <TableRow>
                          <TableCell className={classes.tableCell}>
                            {data.class}
                          </TableCell>
                          <TableCell
                            align="right"
                            className={classes.tableCell}
                          >
                            {confidence}%
                          </TableCell>
                        </TableRow>
                      </TableBody>
                    </Table>
                  </TableContainer>
                </CardContent>
              )}

              {selectedFile && (
                <CardContent>
                  <Typography variant="body2" align="center">
                    {selectedFile.name}
                  </Typography>
                </CardContent>
              )}
            </Card>
          </Grid>

          {selectedFile && (
            <Grid item className={classes.buttonGrid}>
              <ColorButton
                variant="contained"
                className={classes.clearButton}
                size="large"
                onClick={clearData}
                startIcon={<Clear fontSize="large" />}
                disabled={isLoading}
              >
                Clear
              </ColorButton>
            </Grid>
          )}
        </Grid>
      </Container>
    </React.Fragment>
  );
};

export default ImageUpload;